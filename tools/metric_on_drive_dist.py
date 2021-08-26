"""
This file evaluation mAP with regarding the driving distance as the detection range.
Both BBoxes of GT and prediction will be ignored in this evaluation.
Since we assume that all cooperative vehicles can also share their own location when
they are sharing cooperative information, the GT BBoxes of all data-sharing vehicles
are also added to the predictions followed by a nms for the final evaluation.
"""
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import json, tqdm
from ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from utils.batch_statistics import StatsRecorder
from utils.common_utils import limit_period
from skimage.io import imread
from datasets.comap.formatting import read_net, filter_gt_boxes_via_driving_dist

data_root = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m"
pc_range = np.array([-57.6, -57.6, -0.1, 57.6, 57.6, 3.9])
binary_map_file = "/media/hdd/ophelia/fusion_ssd/datasets/binary_map_closed.png"
origBoundary = np.array([-400, -320, 320, 320])
device = torch.device('cuda')
obs_r = pc_range[3]
binary_map = imread(binary_map_file)
sumo_net_points = read_net("/media/hdd/ophelia/fusion_ssd/datasets/Town05.net.xml")


def eval(det_file, IOU_thr=0.7):
    """
    1. remove detections that are not on the roads
    2. ignore correct detections that are not in the driving range
    """
    detection_data = torch.load(det_file)
    samples = detection_data['samples']
    pred_boxes_all = detection_data['pred_boxes']
    gt_official_all = detection_data['gt_boxes']
    confidences = detection_data['confidences']
    ids = detection_data['ids']

    list_sample = []
    list_conf = []
    list_tp = []
    N_gt = 0

    for s in tqdm.tqdm(samples):
        pred_boxes = pred_boxes_all[s]
        vehicle_info_file = os.path.join(data_root, 'poses_global', s + '.txt')
        vehicles_info = np.loadtxt(vehicle_info_file, dtype=str)[:, [0, 2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        vehicles_info_dict = {int(v_info[0]): v_info[1:] for v_info in vehicles_info}
        v_ego = vehicles_info_dict[0]
        # remove bboxes that are not on the road
        xy_min = v_ego[:2] - origBoundary[:2] - np.array([obs_r, obs_r])
        xy_max = xy_min + np.array([obs_r, obs_r]) * 2
        xy_min_inds = np.floor(xy_min / 0.1).astype(np.int)
        xy_max_inds = np.floor(xy_max / 0.1).astype(np.int)
        cur_image = binary_map[xy_min_inds[0]:xy_max_inds[0], xy_min_inds[1]:xy_max_inds[1]]
        indices = np.floor((pred_boxes[:, :2].cpu().numpy() -
                            np.array([-obs_r, -obs_r]).reshape(1, 2)) / 0.1).astype(np.int)
        indices = np.clip(indices, a_min=0, a_max=max(cur_image.shape) - 1)
        # fig = plt.figure(figsize=(5, 5))
        # plt.imshow(cur_image)
        # plt.scatter(indices[:, 1], indices[:, 0], s=20, c='r', marker='s')
        # plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
        # plt.close()
        mask = torch.tensor(cur_image[indices[:, 0], indices[:, 1]], device=device).bool()
        pred_boxes = pred_boxes[mask]
        confs = confidences[s][mask]
        # remove bboxes that are not in driving range
        filtered_boxes, _ = filter_gt_boxes_via_driving_dist(sumo_net_points,
                                                          gt_official_all[s].cpu().numpy(),
                                                          v_ego[:2], s)
        gt_boxes = torch.tensor(filtered_boxes, device=device).float()
        gt_boxes[:, 6] = limit_period(gt_boxes[:, 6], 0.5, 2 * np.pi)
        filtered_boxes, mask = filter_gt_boxes_via_driving_dist(sumo_net_points,
                                                              pred_boxes.cpu().numpy(),
                                                              v_ego[:2], s)
        pred_boxes = torch.tensor(filtered_boxes, device=device).float()
        pred_boxes[:, 6] = limit_period(pred_boxes[:, 6], 0.5, 2 * np.pi)
        confs = confs[mask]

        if len(pred_boxes)>0 and len(gt_boxes)>0:
            ious = boxes_iou3d_gpu(pred_boxes, gt_boxes)
            max_iou_pred_to_gts = ious.max(dim=1)
            max_iou_gt_to_preds = ious.max(dim=0)
            tp = max_iou_pred_to_gts[0] >= IOU_thr
            is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                            == torch.tensor([i for i in range(len(tp))], device=tp.device)
            tp[torch.logical_not(is_best_match)] = False
            list_tp.extend(tp)
            list_conf.extend(confs)
            list_sample.extend([s] * len(tp))
            N_gt += gt_boxes.shape[0]
        elif len(pred_boxes)==0:
            N_gt += gt_boxes.shape[0]
        elif len(gt_boxes[s]) == 0:
            tp = torch.zeros(len(pred_boxes[s]), device=pred_boxes[s].device)
            list_tp.extend(tp.bool())
    order_inds = torch.tensor(list_conf).argsort(descending=True)
    tp_all = torch.tensor(list_tp)[order_inds]
    list_accTP = tp_all.cumsum(dim=0)
    # list_accFP = torch.logical_not(tp_all).cumsum(dim=0)
    prec = list_accTP.float() / torch.arange(1, len(list_sample) + 1)
    rec = list_accTP.float() / N_gt

    mrec = []
    mrec.append(0)
    [mrec.append(e.item()) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e.item()) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


if __name__=="__main__":
    ious = [0.3, 0.4, 0.5, 0.6, 0.7]
    test_path = "/media/hdd/ophelia/koko/experiments-output/fusion-pvrcnn/exp6/test_result_40m_ep10"

    with open(os.path.join(test_path, 'mAP_on_driving_distance.txt'), 'w') as fh:
        for n in range(5):
            result = []
            for iou in ious:
                det_file = os.path.join(test_path, 'thr0.3_ncoop{}.pth'.format(n))
                mAP = eval(det_file, iou)
                print('{:5.2f}'.format(mAP[0] * 100))
                result.append(mAP[0] * 100)
            result_str = ('{:5.2f} ' * 5).format(*result)
            print(result_str)
            fh.write(result_str)
            fh.write('\n')





