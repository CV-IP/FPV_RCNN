import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
pi = 3.141592653

def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class Matcher(nn.Module):
    def __init__(self, pc_range):
        super(Matcher, self).__init__()
        self.pc_range = pc_range

    @torch.no_grad()
    def forward(self, data_dict):
        batch_size = data_dict['batch_size']
        pred_boxes = data_dict['det_boxes_ego_coords']
        pred_scores = data_dict['det_scores']
        # shifts_coop_to_ego = data_dict['translations'][1:] - data_dict['translations'][:1]
        # for b in range(1, batch_size):
        #     pred_boxes[b][:, :2] = pred_boxes[b][:, :2] + shifts_coop_to_ego[b-1][None, :2]
        # coop_boxes_cnt = [len(boxes) for boxes in pred_boxes]

        clusters, scores = self.clustering(pred_boxes, pred_scores)
        data_dict['boxes_fused'] = self.cluster_fusion(clusters, scores)
        # from vlib.point import draw_points_boxes_plt, draw_box_plt
        # for i, c in enumerate(clusters):
        #     ax = plt.subplot(111)
        #     ax.set_aspect('equal', 'box')
        #     draw_box_plt(c.cpu().numpy(), ax, 'b')
        #     draw_box_plt(data_dict['boxes_fused'][i].view(1, 7).cpu().numpy(), ax, 'r')
        #     plt.show()
        #     plt.close()
        return data_dict

    def clustering(self, pred_boxes, pred_scores):
        """
        Assign predicted boxes to clusters according to their ious with each other
        """
        pred_boxes_cat = torch.cat(pred_boxes, dim=0)

        pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
        pred_scores_cat = torch.cat(pred_scores, dim=0)
        ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
        cluster_indices = torch.zeros(len(ious)).int() # gt assignments of preds
        cur_cluster_id = 1
        while torch.any(cluster_indices == 0):
            cur_idx = torch.where(cluster_indices == 0)[0][0] # find the idx of the first pred which is not assigned yet
            cluster_indices[torch.where(ious[cur_idx] > 0.3)[0]] = cur_cluster_id
            cur_cluster_id += 1
        clusters = []
        scores = []
        for i in range(1, cluster_indices.max().item() + 1):
            clusters.append(pred_boxes_cat[cluster_indices==i])
            scores.append(pred_scores_cat[cluster_indices==i])
        if len(scores)==0:
            print('debug')

        return clusters, scores

    def cluster_fusion(self, clusters, scores):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        boxes_fused = []
        for c, s in zip(clusters, scores):
            # reverse direction for non-dominant direction of boxes
            dirs = c[:, -1]
            max_score_idx = torch.argmax(s)
            dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
            lt_pi = (dirs_diff > pi).int()
            dirs_diff = dirs_diff * (1 - lt_pi) + (2 * pi - dirs_diff) * lt_pi
            score_lt_half_pi = s[dirs_diff > pi / 2].sum() # larger than
            score_set_half_pi = s[dirs_diff <= pi / 2].sum() # small equal than
            # select larger scored direction as final direction
            if score_lt_half_pi <= score_set_half_pi:
                dirs[dirs_diff > pi / 2] += pi
            else:
                dirs[dirs_diff <= pi / 2] += pi
            dirs = limit_period(dirs)
            s_normalized = s / s.sum()
            sint = torch.sin(dirs) * s_normalized
            cost = torch.cos(dirs) * s_normalized
            theta = torch.atan2(sint.sum(), cost.sum()).view(1,)
            center_dim = c[:, :-1] * s_normalized[:, None]
            boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
        if len(boxes_fused) > 0:
            boxes_fused = torch.stack(boxes_fused, dim=0)
        else:
            boxes_fused = None
            print('debug')
        return boxes_fused
