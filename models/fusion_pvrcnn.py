from torch import nn
from torch.nn import Sequential
import torch
from models import *
import numpy as np
from models.utils import xavier_init

from matplotlib import pyplot as plt
from vlib.image import draw_box_plt


def _build_deconv_block(in_channels, out_channels):
    return [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()]


def _build_conv_block(in_channels, out_channels, stride=1):
    return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()]


class FusionPVRCNN(nn.Module):
    """
    This model is based on CIA-SSD. Each point cloud will be forwarded once to obtain
    logits features, which are then fused to refine the object detection.
    """
    def __init__(self, mcfg, cfg, dcfg):
        super(FusionPVRCNN, self).__init__()
        self.mcfg = mcfg
        self.test_cfg = cfg.TEST
        self.pc_range = dcfg.pc_range

        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.ssfa = SSFA(mcfg.SSFA)
        self.head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)

        self.vsa = VoxelSetAbstraction(mcfg.VSA, dcfg.voxel_size, dcfg.pc_range, num_bev_features=128,
                                       num_rawpoint_features=3)
        self.matcher = Matcher(dcfg.pc_range)
        self.roi_head = RoIHead(mcfg.ROI_HEAD, self.head.box_coder)

        # self.set_trainable_parameters(mcfg.params_train)

    def set_trainable_parameters(self, block_names):
        for param in self.named_parameters():
            m = getattr(self, param[0].split('.')[0])
            if m.__class__.__name__ not in block_names:
                param[1].requires_grad = False

    def forward(self, batch_dict):
        n_coop = len(batch_dict['ids']) - 1
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        logits_feature = self.ssfa(batch_dict['spatial_features'])
        preds = self.head(logits_feature)
        batch_dict["preds_egos"] = preds

        if n_coop > 0:
            batch_dict, num_total_dets = self.shift_and_filter_preds(batch_dict)

            # # debug visualization
            # vis_points = batch_dict['points'].cpu().numpy()
            # vis_points = vis_points[vis_points[:, 0]==0, 1:]
            # vis_boxes_list = [boxes.cpu().numpy() for boxes in batch_dict['det_boxes_ego_coords']]
            # plt.figure(figsize=(10, 10))
            # ax = plt.subplot(111)
            # ax.axis('equal')
            # ax.plot(vis_points[:, 0], vis_points[:, 1], '.y', markersize=0.3)
            # colors = ['r', 'b', 'c', 'g', 'm']
            # for i, vis_boxes in enumerate(vis_boxes_list):
            #     ax = draw_box_plt(vis_boxes, ax, color=colors[i])
            # plt.savefig('/media/hdd/ophelia/koko/experiments-output/fusion-pvrcnn/exp1/preds_filtered.png')
            # plt.close()

            if num_total_dets > 0:
                batch_dict = self.vsa(batch_dict)
                # for i, vis_kpt in enumerate(batch_dict['point_coords']):
                #     vis_kpt = vis_kpt.cpu().numpy()
                #     plt.plot(vis_kpt[:, 0], vis_kpt[:, 1], '.', markersize=1, color=colors[i])
                # plt.savefig('/media/hdd/ophelia/koko/experiments-output/fusion-pvrcnn/exp1/kpts.png')
                # plt.close()
                batch_dict = self.matcher(batch_dict)
                batch_dict = self.roi_head(batch_dict)

        return batch_dict

    def shift_and_filter_preds(self, batch_dict):
        batch_dict['det_boxes'] = []
        batch_dict['det_scores'] = []
        batch_dict['det_boxes_ego_coords'] = []
        batch_dets = self.post_processing_ego(batch_dict, self.test_cfg, det_all=True)
        shifts_coop_to_ego = batch_dict['translations']- batch_dict['translations'][:1]
        total_dets = 0
        for b, dets in enumerate(batch_dets):
            pred_boxes = dets['box_lidar']
            pred_boxes_ego_coords = pred_boxes.clone().detach()
            pred_boxes_ego_coords[:, :2] = pred_boxes_ego_coords[:, :2] + shifts_coop_to_ego[b][None, :2]
            # mask pred. boxes that are in the detection range
            in_range_mask = torch.norm(pred_boxes_ego_coords[:, :2], dim=-1) < self.pc_range[3]
            batch_dict['det_boxes'].append(pred_boxes[in_range_mask])
            batch_dict['det_scores'].append(dets['scores'][in_range_mask])
            batch_dict['det_boxes_ego_coords'].append(pred_boxes_ego_coords[in_range_mask])
            total_dets += in_range_mask.sum()
        return batch_dict, total_dets

    def _make_model_input(self, batch_dict, batch_idx):
        data_dict = {}
        for k, v in batch_dict.items():
            if isinstance(v, list):
                data_dict[k] = v[batch_idx]
            elif k in ["anchors", "labels", "reg_targets", "reg_weights"]:
                data_dict[k] = v[batch_idx].unsqueeze(dim=0)
            else:
                data_dict[k] = v
        data_dict['batch_size'] = len(data_dict['cloud_sizes'])
        return data_dict

    def loss(self, batch_dict):
        batch_dict['preds_dict'] = batch_dict['preds_egos']
        loss_ego = self.head.loss(batch_dict)

        # supervise fusion detetion only for ego vehicle
        if len(batch_dict['ids']) > 1 and 'boxes_fused' in batch_dict:
            loss_fuse = self.roi_head.get_loss(batch_dict)
            loss = {
                'loss': loss_ego['loss'] + loss_fuse['loss'],
                'loss_ego': loss_ego['loss'],
                'loss_fuse': loss_fuse['loss'],
                'loss_fuse_cls': loss_fuse['loss_cls'],
                'loss_fuse_reg': loss_fuse['loss_reg'],
            }

        else:
            loss = {
                'loss': loss_ego['loss'],
                'loss_ego': loss_ego['loss'],
            }

        return loss

    def post_processing(self, batch_dict, test_cfg):
        if 'rcnn_reg' in batch_dict.keys():
            detections = self.roi_head.get_detections(batch_dict)
        elif 'preds_final' in batch_dict.keys():
            detections = self.post_processing_final(batch_dict, test_cfg)
        else:
            detections = self.post_processing_ego(batch_dict, test_cfg)
        return detections

    def post_processing_final(self, batch_dict, test_cfg):
        preds_dict = batch_dict['preds_final']
        anchors = batch_dict["anchors"][0:1]
        batch_size = 1
        anchors_flattened = anchors.view(batch_size, -1, self.head.box_n_dim)
        cls_preds = preds_dict["cls_preds"].view(batch_size, -1, self.head.num_classes)  # [8, 70400, 1]
        reg_preds = preds_dict["box_preds"].view(batch_size, -1, self.head.box_coder.code_size)  # [batch_size, 70400, 7]
        iou_preds = preds_dict["iou_preds"].view(batch_size, -1, 1)
        coop_boxes = batch_dict["coop_boxes_in_egoCS"][0:1].view(batch_size, -1, self.head.box_coder.code_size)
        if self.head.use_direction_classifier:
            dir_preds = preds_dict["dir_cls_preds"].view(batch_size, -1, 2)
        else:
            dir_preds = None

        box_preds = self.head.box_coder.decode_torch(reg_preds[:, :, :self.head.box_coder.code_size],
                                                anchors_flattened)# .squeeze()
        detections = self.fusion_detect.det_head.get_task_detections(test_cfg,
                                              cls_preds, box_preds,
                                              dir_preds, iou_preds,
                                              batch_coop_boxes=coop_boxes,
                                              batch_anchors=anchors)
        return detections

    def post_processing_ego(self, batch_dict, test_cfg, det_all=False):
        preds_dict = batch_dict['preds_egos']
        anchors = batch_dict["anchors"]
        coop_boxes = None
        batch_size = 1
        if det_all:
            batch_size = batch_dict['batch_size']
        else:
            anchors = anchors[0:1]
            if self.training:
                coop_boxes = batch_dict["coop_boxes_in_egoCS"][:batch_size].view(batch_size, -1,
                                                                             self.head.box_coder.code_size)
        anchors_flattened = anchors.view(batch_size, -1, self.head.box_n_dim)
        cls_preds = preds_dict["cls_preds"][:batch_size].view(batch_size, -1, self.head.num_classes)  # [8, 70400, 1]
        reg_preds = preds_dict["box_preds"][:batch_size].view(batch_size, -1, self.head.box_coder.code_size)  # [batch_size, 70400, 7]
        iou_preds = preds_dict["iou_preds"][:batch_size].view(batch_size, -1, 1)

        if self.head.use_direction_classifier:
            dir_preds = preds_dict["dir_cls_preds"][:batch_size].view(batch_size, -1, 2)
        else:
            dir_preds = None

        box_preds = self.head.box_coder.decode_torch(reg_preds[:, :, :self.head.box_coder.code_size],
                                                anchors_flattened)# .squeeze()
        detections = self.head.get_task_detections(test_cfg,
                                              cls_preds, box_preds,
                                              dir_preds, iou_preds,
                                              batch_coop_boxes=coop_boxes,
                                              batch_anchors=anchors)

        return detections


class CPMEnc(nn.Module):
    """Collective Perception Message encoding module"""
    def __init__(self, in_channel, out_channel, n_layers=2, upsample=0, **kwargs):
        super(CPMEnc, self).__init__()
        if 'encode_feature' in kwargs:
            self.encode_feature = kwargs['encode_feature']
        cur_channels_in = in_channel
        cur_channels_out = out_channel

        assert n_layers>upsample
        block = []

        for i in range(n_layers - 1):
            cur_channels_out = cur_channels_in // 2 if cur_channels_in > 2*out_channel else out_channel
            conv_fn = _build_deconv_block if i<upsample else _build_conv_block
            block.extend(conv_fn(cur_channels_in, cur_channels_out))
            cur_channels_in = cur_channels_out
        self.encoder = Sequential(*block)

        self.conv_out = Sequential(*_build_conv_block(cur_channels_out, out_channel))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        x = self.encoder(x)
        out = self.conv_out(x)

        return out


class FUDET(nn.Module):
    def __init__(self, mcfg, dcfg):
        super(FUDET, self).__init__()
        self.fusion_res = mcfg.FUDET['fusion_resolution']
        self.fusion_score = mcfg.FUDET['fusion_score']

        feature_dim = mcfg.CPMEnc['out_channel']
        det_dim = mcfg.CPMEnc['in_channel']

        block_up = []
        cur_in = feature_dim
        for c in mcfg.FUDET['upsample_channels']:
            block_up.extend(_build_deconv_block(cur_in, c))
            cur_in = c
        self.up_features = Sequential(*block_up)
        self.up_weights = nn.Conv2d(cur_in, 1, kernel_size=1, bias=False)

        block_down = []
        # cur_in = cur_in * 2
        for c in mcfg.FUDET['downsample_channels']:
            block_down.extend(_build_conv_block(cur_in, c, stride=2))
            cur_in = c
        self.convs_fuse = Sequential(*block_down)

        convs = []
        for c in mcfg.FUDET['conv_head_channels']:
            convs.extend(_build_conv_block(cur_in, c))
            cur_in = c
        self.convs_head = Sequential(*convs)
        self.det_head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, batch_dict):
        feature_fused = self.fusion(batch_dict)
        out = self.convs_fuse(feature_fused)
        out = self.convs_head(out)
        batch_dict['fused_features'] = out
        out = self.det_head(out)
        batch_dict['preds_final'] = out
        return batch_dict

    def fusion(self, data_dict):
        cpm_features = data_dict['cpm_features']
        feat_mask = torch.sigmoid(data_dict['preds_egos']['cls_preds']).sum(dim=-1) > self.fusion_score
        masked_cpm_features = cpm_features
        masked_cpm_features[1:] = masked_cpm_features[1:] * feat_mask.unsqueeze(1)[1:] # ego vehicle has full information
        features = self.up_features(masked_cpm_features)
        weights = self.up_weights(features)
        features_ego, features_coop = features[0:1], features[1:]
        weights_coop = weights[1:]

        # shift keypoints to ego coordinate sytem
        translations = data_dict['translations']
        shifts_coop_to_ego = ((translations[1:] - translations[0:1]) / self.fusion_res).int()
        shifted_cpm_features_coop = self._shift2d(features_coop, shifts_coop_to_ego)
        shifted_cpm_weights_coop = self._shift2d(weights_coop, shifts_coop_to_ego)

        # fusion
        cpm_weights_coop_norm = shifted_cpm_weights_coop.softmax(dim=0)
        coop_features_fused = (shifted_cpm_features_coop * cpm_weights_coop_norm).sum(dim=0)
        features_add = features_ego + coop_features_fused
        # features_cat = torch.cat([features_ego, coop_features_fused], dim=0)

        return features_add

    def _shift2d(self, matrices, shifts):
        matrices_out = []
        for mat, shift in zip(matrices, shifts):
            dx, dy = int(shift[1]), int(shift[0])
            shifted_mat = torch.roll(mat, dx, 1)
            if dx < 0:
                shifted_mat[:, dx:, :] = 0
            elif dx > 0:
                shifted_mat[:, 0:dx, :] = 0
            shifted_mat = torch.roll(shifted_mat, dy, 2)
            if dy < 0:
                shifted_mat[:, :, dy:] = 0
            elif dy > 0:
                shifted_mat[:, :, 0:dy] = 0
            matrices_out.append(shifted_mat)
        return torch.stack(matrices_out, dim=0)



# sx_max = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), weights.shape[2])
# sy_max = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), weights.shape[3])
# x_max = torch.min(sx_max, weights.shape[2] + shifts_coop_to_ego[:, 0])
# y_max = torch.min(sy_max, weights.shape[3] + shifts_coop_to_ego[:, 1])
# sx_min = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), 0)
# sy_min = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), 0)
# x_min = torch.max(sx_min, shifts_coop_to_ego[:, 0])
# y_min = torch.max(sy_min, shifts_coop_to_ego[:, 1])
# xs = xs[inds > 0] + shifts_coop_to_ego[inds[inds > 0], 0]
# ys = ys[inds > 0] + shifts_coop_to_ego[inds[inds > 0], 1]

