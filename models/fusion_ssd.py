from torch import nn
from torch.nn import Sequential
import torch
from models import *
import numpy as np
from models.utils import xavier_init
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


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


class FusionSSD(nn.Module):
    """
    This model is based on CIA-SSD. Each point cloud will be forwarded once to obtain
    logits features, which are then fused to refine the object detection.
    """
    def __init__(self, mcfg, cfg, dcfg):
        super(FusionSSD, self).__init__()
        self.mcfg = mcfg
        self.test_cfg = cfg.TEST
        self.pc_range = dcfg.pc_range
        self.cpm_feature_size = np.array(dcfg.feature_map_size)[1:] * 2** len(mcfg.FUDET['upsample_channels'])
        self.cpm_feature_reso = dcfg.grid_size[:2] / self.cpm_feature_size * np.array(dcfg.voxel_size)[:2]

        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.ssfa = SSFA(mcfg.SSFA)
        self.cpm_enc = CPMEnc(**mcfg.CPMEnc)
        self.head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)
        self.fusion_detect = FUDET(mcfg, dcfg)
        # self.vsa = VoxelSetAbstraction(mcfg.VSA, dcfg.voxel_size, dcfg.pc_range, num_bev_features=128,
        #                                num_rawpoint_features=3)

        # self.set_trainable_parameters(mcfg.params_train)

    def set_trainable_parameters(self, block_names):
        for param in self.named_parameters():
            m = getattr(self, param[0].split('.')[0])
            if m.__class__.__name__ not in block_names:
                param[1].requires_grad = False

    def forward(self, batch_dict):
        n_coop = len(batch_dict['ids']) - 1
        data_dict = self.vfe(batch_dict)
        data_dict = self.spconv_block(data_dict)
        data_dict = self.map_to_bev(data_dict)
        logits_feature = self.ssfa(data_dict['spatial_features'])
        preds = self.head(logits_feature)
        batch_dict["preds_egos"] = preds
        batch_dict["detections"] = self.post_processing_ego(batch_dict, self.test_cfg, det_all=True)

        if n_coop > 0:
            if self.cpm_enc.encode_feature == 'spconv':
                encode_feature = data_dict['spatial_features']
            elif self.cpm_enc.encode_feature == 'ssfa':
                encode_feature = logits_feature
            # elif self.cpm_enc.encode_feature == 'keypoints':
            #     batch_dict['det_boxes'] = []
            #     batch_dict['det_scores'] = []
            #     for dets in self.post_processing_ego(data_dict,
            #                                     self.test_cfg, det_all=True):
            #
            #         batch_dict['det_boxes'].append(dets['box_lidar'])
            #         batch_dict['det_scores'].append(dets['scores'])
            #
            #     batch_dict = self.vsa(batch_dict)

            batch_dict["cpm_features"] = self.cpm_enc(encode_feature)

            batch_dict = self.fusion_detect(batch_dict)

        return batch_dict

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
        # batch_preds = batch_dict['batch_preds']
        # preds_ego = {
        #     'box_preds': torch.stack([pred['box_preds'][0] for pred in batch_preds], dim=0),
        #     'cls_preds': torch.stack([pred['cls_preds'][0] for pred in batch_preds], dim=0),
        #     'dir_cls_preds': torch.stack([pred['dir_cls_preds'][0] for pred in batch_preds], dim=0),
        #     'iou_preds': torch.stack([pred['iou_preds'][0] for pred in batch_preds], dim=0),
        #     'var_box_preds': torch.stack([pred['var_box_preds'][0] for pred in batch_preds], dim=0),
        # }
        batch_dict['preds_dict'] = batch_dict['preds_egos']
        loss_ego = self.head.loss(batch_dict)
        # supervise fusion detetion only for ego vehicle
        if 'preds_final' in batch_dict.keys():
            target_fused = batch_dict["target_fused"]
            batch_dict['preds_dict'] = batch_dict['preds_final']
            batch_dict['reg_targets'] = target_fused['reg_targets']
            batch_dict['reg_weights'] = target_fused['reg_weights']
            batch_dict['anchors'] = batch_dict['anchors'][0:1]
            batch_dict['labels'] = target_fused['labels']
            loss = self.fusion_detect.det_head.loss(batch_dict)

            loss.update({
                'loss_ego': loss_ego['loss'],
                'loss_fuse': loss['loss'],
                'loss': loss_ego['loss'] + loss['loss']
            })
        else:
            loss = loss_ego
            loss.update({
                'loss_ego': loss_ego['loss'],
                'loss_fuse': 0,
                'loss': loss_ego['loss']
            })
        if hasattr(self.mcfg, 'RDFT') and self.mcfg.RDFT['use']:
            loss_rdf = self.deep_coral_loss(batch_dict['fused_features'], batch_dict['rdf_features'],
                                              batch_dict['target_fused']['reg_weights'] > 0)
            loss.update({
                'loss_rdf': loss_rdf,
                'loss': loss['loss'] + loss_rdf
            })

        # ######plot#####
        # import matplotlib.pyplot as plt
        # from vlib.point import draw_box_plt
        # predictions_dicts = self.head.post_processing(batch_dict, self.test_cfg)
        # pred_boxes = [pred_dict['box_lidar'] for pred_dict in predictions_dicts]
        # ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
        # ax.set_aspect('equal', 'box')
        # ax.set(xlim=(self.pc_range[0], self.pc_range[3]),
        #        ylim=(self.pc_range[1], self.pc_range[4]))
        # points = batch_dict['points'][0][0]
        # ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
        # ax = draw_box_plt(batch_dict['gt_boxes'][0], ax, color='green', linewidth_scale=2)
        # ax = draw_box_plt(pred_boxes[0], ax, color='red')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        # plt.close()
        # ##############

        return loss

    def post_processing(self, batch_dict, test_cfg):
        if 'preds_final' in batch_dict.keys():
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
        self.pc_range = dcfg.pc_range

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
        # feat_mask_s = torch.sigmoid(data_dict['preds_egos']['cls_preds']).sum(dim=-1) > self.fusion_score
        feat_mask = self.get_feat_mask(data_dict['detections'], cpm_features.permute(1, 0, 2, 3).shape[1:])
        data_dict['feature_mask'] = feat_mask
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(feat_mask_s[0].cpu().numpy())
        # plt.subplot(122)
        # plt.imshow(feat_mask[0].cpu().numpy())
        # plt.show()
        # plt.close()
        masked_cpm_features = cpm_features
        masked_cpm_features[1:] = masked_cpm_features[1:] * feat_mask.unsqueeze(1)[1:] # ego vehicle has full information
        features = self.up_features(masked_cpm_features)
        data_dict['decoded_cpm_features'] = features
        weights = self.up_weights(features).sigmoid()
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

    def get_feat_mask(self, detections, out_shape):
        max_len = max([len(dets['box_lidar']) for dets in detections])
        batch_dets = detections[0]['box_lidar'].new_zeros((len(detections), max_len, 7))
        for i, dets in enumerate(detections):
            boxes = dets['box_lidar']
            batch_dets[i, :len(boxes)] = boxes
        batch_dets[:, :, 2] = 0
        x = torch.arange(self.pc_range[0] + 0.4, self.pc_range[3], 0.8).to(batch_dets.device)
        yy, xx = torch.meshgrid(x, x)
        points = batch_dets.new_zeros((len(xx.reshape(-1)), 3))
        points[:, 0] = xx.reshape(-1)
        points[:, 1] = yy.reshape(-1)
        batch_points = torch.stack([points] * len(detections), dim=0)
        box_idxs_of_pts = points_in_boxes_gpu(batch_points, batch_dets).view(out_shape) >= 0

        return box_idxs_of_pts

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

