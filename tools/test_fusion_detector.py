import copy

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from vlib.image import draw_box_plt
from utils.train_utils import *
from models.utils import load_model_dict
import gzip


def test_net(cfgs):
    dcfg, mcfg, cfg = cfgs
    n_coop = cfg.TEST['n_coop'] if 'n_coop' in list(cfg.TEST.keys()) else 0
    com_range = cfg.TEST['com_range'] if 'com_range' in list(cfg.TEST.keys()) else 0
    print("Building test dataloader...")
    test_dataloader = build_dataloader(dcfg, cfg, train=False)
    # use this block if only test one frame
    frame = "53_000646"
    idx = np.where(np.array(test_dataloader.dataset.file_list) == frame)[0]
    test_dataloader.dataset.file_list = [frame]
    test_dataloader.dataset.coop_files = [test_dataloader.dataset.coop_files[int(idx)]]
    print("\bfinished.")
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("Building model...")
    model = build_model(mcfg, cfg, dcfg).to(device)
    print("\bfinished.")

    log_path = Path(cfg.PATHS['run'])
    if os.path.exists(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['total_epochs']))):
        test_out_path = log_path / 'test_result_{:2d}m_ep{:2d}'.format(com_range, cfg.TRAIN['total_epochs'])
        ckpt_path = str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['total_epochs']))
    else:
        test_out_path = log_path / 'test_result_{:2d}m_latest'.format(com_range)
        ckpt_path = str(log_path / 'latest.pth')
    test_out_path.mkdir(exist_ok=True)

    # get metric
    from eval.mAP import MetricAP
    metric = MetricAP(cfg.TEST, test_out_path, device='cuda', bev=cfg.TEST['bev'])
    thrs = cfg.TEST['ap_ious']

    # if metric.has_test_detections:
    #     aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    #     with open(test_out_path / 'thr{}_ncoop{}.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
    #         for thr,  ap in zip(thrs, aps):
    #             fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))
    #     return

    # load checkpoint
    pretrained_dict = torch.load(ckpt_path)
    model = load_model_dict(model, pretrained_dict)

    # dir for save test images
    if cfg.TEST['save_img']:
        images_path = (test_out_path / 'images_{}_{}'.format(cfg.TEST['score_threshold'], n_coop))
        images_path.mkdir(exist_ok=True)
    # dir for save cpms
    cpms_path = (test_out_path / 'cpms_{}_{}'.format(cfg.TEST['score_threshold'], n_coop))
    cpms_path.mkdir(exist_ok=True)
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    model.eval()
    direcs = []
    i = 0
    with torch.no_grad():
        print("Start testing")
        for batch_data in tqdm(test_dataloader):
            i += 1
            if 'cloud_fused' not in batch_data:
                points = batch_data['points']
                points = [points[points[:, 0]==i, 1:] for i in range(batch_data['batch_size'])]
                translations = batch_data['translations']
                translations = translations - translations[0:1]
                points = np.concatenate([pts + t for pts, t in zip(points, translations)], axis=0)
            else:
                points = batch_data.pop('cloud_fused')

            boxes = batch_data['gt_boxes'][0:1]
            load_data_to_device(batch_data)

            # Forward pass
            batch_data = model(batch_data)
            # Save CPMs
            if mcfg.name=='fusion_pvrcnn' and cfg.TEST['n_coop'] > 0:
                ids = batch_data['ids']
                point_features = batch_data['point_features']
                point_coords = batch_data['point_coords'] # in ego coords. frame
                cur_path = str(cpms_path / batch_data['frame'])
                os.makedirs(cur_path, exist_ok=True)
                for id, coords, features in zip(ids, point_coords, point_features):
                    data = torch.cat([coords, features], dim=1).cpu().numpy()
                    with open(os.path.join(cur_path, "{:06d}".format(id)), 'wb') as f:
                        f.write(data)
            elif mcfg.name=='fusion_ssd' and cfg.TEST['n_coop'] > 0:
                ids = batch_data['ids']
                spatial_features = batch_data['spatial_features']
                cpm_features = batch_data['cpm_features']
                dec_cpm_features = batch_data['decoded_cpm_features']
                feat_mask = batch_data['feature_mask']
                cur_path = str(cpms_path / batch_data['frame'])
                os.makedirs(cur_path, exist_ok=True)
                for id, spf, cpmf, cpmdf, mask in zip(ids, spatial_features, cpm_features, dec_cpm_features, feat_mask):
                    np.savez_compressed(os.path.join(cur_path, "{:06d}.npz".format(id)),
                                        sp=spf.permute(1, 2, 0).cpu().numpy(),
                                        cpm=cpmf.permute(1, 2, 0).cpu().numpy(),
                                        dcpm=cpmdf.permute(1, 2, 0).cpu().numpy(),
                                        mask=mask.cpu().numpy())

            predictions_dicts = model.post_processing(batch_data, cfg.TEST)
            pred_boxes = [pred_dict['box_lidar'] for pred_dict in predictions_dicts]
            scores = [pred_dict['scores'] for pred_dict in predictions_dicts]
            direcs.extend([boxes[:, -1] for boxes in pred_boxes])
            if cfg.TEST['save_img'] and i % 1 == 0:
                ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(dcfg.pc_range[0], dcfg.pc_range[3]),
                       ylim=(dcfg.pc_range[1], dcfg.pc_range[4]))
                ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
                ax = draw_box_plt(boxes[0], ax, color='green', linewidth_scale=2)
                ax = draw_box_plt(pred_boxes[0], ax, color='red')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(str(images_path / '{}.png'.format(batch_data['frame'])))
                plt.close()

            metric.add_samples([batch_data['frame']], pred_boxes, batch_data['gt_boxes'], scores,
                               ids=[batch_data['ids']])

    metric.save_detections()
    aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    with open(test_out_path / 'thr{}_ncoop{}.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
        for thr, ap in zip(thrs, aps):
            fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))


def test_fusion_pvrcnn_group():
    cfgs = cfg_from_py("fusion_ssd_comap")
    num_kpt = [1024, 2048]
    num_ch = [32, 64, 128]
    run_path = '/media/hdd/ophelia/koko/experiments-output/fusion-pvrcnn'
    for i, nk in enumerate(num_kpt):
        for j, nc in enumerate(num_ch):
            print("################{}, {}################".format(nk, nc))
            dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
            mcfg.VSA['num_keypoints'] = nk
            mcfg.VSA['num_out_features'] = nc
            mcfg.ROI_HEAD['input_channels'] = nc
            cur_test_path = os.path.join(run_path, "{:d}ch{:d}".format(nk, nc))
            cfg.PATHS['run'] = cur_test_path
            print(cur_test_path)
            for n in [0, 1, 2, 3, 4]:
                # dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
                cfg.TEST['n_coop'] = n
                # print('debug')
                test_net(copy.deepcopy((dcfg, mcfg, cfg))) # some parameter might be modyfied, for example mlps


if __name__=="__main__":
    cfgs = cfg_from_py("fusion_ssd_comap")
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
    cfg.PATHS['run'] = '/media/hdd/ophelia/koko/experiments-output/fusion-ssd/exp13'
    for n in [2]: #[0, 1, 2, 3, 4]:
        # dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
        cfg.TEST['n_coop'] = n
        # print('debug')
        test_net(copy.deepcopy((dcfg, mcfg, cfg)))