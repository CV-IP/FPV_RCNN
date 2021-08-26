import copy
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils.points_utils import add_gps_noise_bev
from utils.box_np_ops import limit_period
from utils.tfs import global_rotation, rotate_points_along_z
from glob import glob


class CoMapDataset(Dataset):
    def __init__(self, cfg, training=True, n_coop='random>0', com_range=60):
        super().__init__()
        self.training = training
        self.n_coop = n_coop
        self.com_range = com_range
        self.cfg = cfg
        self.root = Path(self.cfg.root)
        self.add_gps_noise = cfg.add_gps_noise
        self.gps_noise_std = cfg.gps_noise_std
        if not self.training and cfg.node_selection_mode is not None:
            self.node_selection = np.load(os.path.join(cfg.root, cfg.node_selection_mode + '.npy'),
                                          allow_pickle=True).item()
        else:
            self.node_selection = None
        self.fuse_raw_data = cfg.fuse_raw_data
        if self.fuse_raw_data:
            assert 'egoCS' in cfg.coop_cloud_name # Use clouds in ego lidar coordinate system
        if not Path(self.cfg.root + "/train_val.txt").exists():
            split(cfg)
        if self.mode=="train":
            with open(self.cfg.root + "/train_val.txt", "r") as f:
                self.file_list = f.read().splitlines()
        else:
            with open(self.cfg.root + "/test.txt", "r") as f:
                self.file_list = f.read().splitlines()

        self.coop_files = self.update_file_list()
        str = 'train val set: ' if self.training else 'test set: '
        print(str, len(self.file_list))

        # point label map, 0 as non-relevant classes
        self.label_color_map = {}
        for i, classes in self.cfg.classes.items():
            self.label_color_map[i] = [list(self.cfg.LABEL_COLORS.keys()).index(c) for c in classes]

        self.augmentor = None
        if self.training:
            if getattr(self.cfg, "AUGMENTOR", None):
                from datasets.comap.augmentor import Augmentor
                self.augmentor = Augmentor(cfg)
        
        if hasattr(self.cfg, "TARGET_ASSIGNER"):
            from datasets import assign_target
            self.target_assigner = assign_target.AssignTarget(cfg.pc_range,
                                                              cfg=self.cfg.TARGET_ASSIGNER,
                                                              mode=self.mode)
            
        process_fn = self.cfg.process_fn["train"] if self.training else self.cfg.process_fn["test"]
        self.processors = []
        for fn in process_fn:
            if fn=="points_to_voxel":
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.cfg.voxel_size,
                    point_cloud_range=self.cfg.pc_range,
                    max_num_points=self.cfg.max_points_per_voxel,
                    max_voxels=self.cfg.max_num_voxels
                )
            self.processors.append(getattr(self, fn))
        if self.cfg.BOX_CODER['type'] == 'GroundBoxBevGridCoder':
            from models.box_coders import GroundBoxBevGridCoder
            self.box_coder = GroundBoxBevGridCoder(**self.cfg.BOX_CODER)

    @property
    def mode(self):
        return "train" if self.training else "test"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_dict = self.load_data(index)
        if self.augmentor is not None:
            data_dict = self.augmentor.forward(data_dict)
        for processor in self.processors:
            data_dict = processor(data_dict)

        return self.drop_intermidiate_data(data_dict)

    def update_file_list(self):
        """
        Update file list according to number of cooperative vehicles, frames will be removed if
        the number of cooperative vehicles is less than the given number
        """
        coop_files_list = []
        selected = []
        if self.training or self.n_coop=='random' or self.n_coop==0:
            coop_files_list = [glob(os.path.join(self.cfg.root, self.cfg.coop_cloud_name, file, '*.bin'))
                               for file in self.file_list]
        elif self.n_coop=='random>0':
            for file in self.file_list:
                coop_files = glob(os.path.join(self.cfg.root, self.cfg.coop_cloud_name, file, '*.bin'))
                if len(coop_files)>0: # take samples which has enough clouds to be choosen self.n_coop
                    selected.append(file)
                    coop_files_list.append(coop_files)
            self.file_list = selected
        else:
            for file in self.file_list:
                coop_files = glob(os.path.join(self.cfg.root, self.cfg.coop_cloud_name, file, '*.bin'))
                if self.com_range < 60:
                    tfs = np.load(os.path.join(self.cfg.root, "tfs", file + '.npy'), allow_pickle=True).item()
                    coop_ids = [file.rsplit("/")[-1][:-4] for file in coop_files]
                    ego_loc = tfs['tf_ego'][0:2, -1]
                    coop_locs = [tfs[coop_id][0:2, -1] for coop_id in coop_ids]
                    coop_in_com_range_inds = np.linalg.norm(np.array(coop_locs) - np.array(ego_loc), axis=1) < self.com_range
                    coop_files = np.array(coop_files)[coop_in_com_range_inds].tolist()
                if len(coop_files)>=self.n_coop: # take samples which has enough clouds to be choosen self.n_coop
                    selected.append(file)
                    coop_files_list.append(coop_files)
            self.file_list = selected
        return coop_files_list

    def load_data(self, index):
        # load ego point cloud and gt
        cloud_filename_ego = os.path.join(self.cfg.root, self.cfg.ego_cloud_name,
                                      self.file_list[index] + ".bin")
        cloud_ego = np.fromfile(cloud_filename_ego, dtype="float32").reshape(-1, 4)[:, :3]
        vehicle_info_file = cloud_filename_ego.replace('cloud_ego', 'poses_global').replace('bin', 'txt')
        vehicles_info = np.loadtxt(vehicle_info_file, dtype=str)[:, [0, 2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        vehicles_info_dict = {int(v_info[0]): v_info[1:] for v_info in vehicles_info}
        gt_boxes = []
        cloud_ego = rotate_points_along_z(cloud_ego, vehicles_info_dict[0][-1])
        # shift ego cloud to the global height
        cloud_ego[:, 2] = cloud_ego[:, 2] + vehicles_info_dict[0][2] + vehicles_info_dict[0][-2]/2 + 0.3
        clouds = [cloud_ego]
        cloud_fused = [cloud_ego]
        ids = [0]
        translations = [vehicles_info_dict[0][:3]]
        bbox_filename = os.path.join(self.cfg.root, "bbox_global_z", self.file_list[index], "000000.txt")
        boxes = np.loadtxt(bbox_filename, dtype=np.float).reshape(-1, 7)
        boxes[:, 6] = limit_period(boxes[:, 6], 0.5, 2 * np.pi)
        gt_boxes.append(boxes)

        if not self.n_coop==0:
            # randomly load cooperative clouds
            coop_files = self.coop_files[index]
            if not self.training and self.node_selection is not None:
                coop_nodes = self.node_selection[self.file_list[index]][self.n_coop - 1]
                coop_nodes_mask = np.array(list(map(lambda f: f.rsplit("/")[-1][:-4] in coop_nodes, coop_files)))
                selected = np.where(coop_nodes_mask)[0]
            elif isinstance(self.n_coop, str) and 'random' in self.n_coop:
                n_min = 1 if '>' in self.n_coop else 0
                selected = np.random.choice(list(np.arange(0, len(coop_files))),
                                            np.random.randint(n_min, min(len(coop_files), 4) + 1), replace=False)
            else:
                selected = np.random.choice(list(np.arange(0, len(coop_files))),
                                            self.n_coop, replace=False)
            for cf in selected.tolist():
                cloud_coop = np.fromfile(coop_files[cf], dtype="float32").reshape(-1, 3)
                coop_id = coop_files[cf].split('/')[-1][:-4]
                ids.append(int(coop_id))
                v_info = vehicles_info_dict[int(coop_id)]
                cloud_coop = rotate_points_along_z(cloud_coop, v_info[-1])
                # shift coop cloud to the global height
                cloud_coop[:, 2] = cloud_coop[:, 2] + v_info[2] + v_info[-2] / 2 + 0.3
                # transform xy of coop clouds from body frame to ego frame
                cloud_coop_tf = copy.deepcopy(cloud_coop)
                cloud_coop_tf[:, :2] = cloud_coop[:, :2] + v_info[None, :2] - vehicles_info_dict[0][None, :2]
                if self.add_gps_noise:
                    cloud_coop = add_gps_noise_bev(cloud_coop, self.gps_noise_std)
                    cloud_coop_tf = add_gps_noise_bev(cloud_coop_tf, self.gps_noise_std)
                clouds.append(cloud_coop[:, :3])
                cloud_fused.append(cloud_coop_tf[:, :3])
                translations.append(v_info[:3])
                bbox_filename = os.path.join(self.cfg.root, "bbox_global_z", self.file_list[index],
                                             "{:06d}.txt".format(int(coop_id)))
                boxes = np.loadtxt(bbox_filename, dtype=np.float).reshape(-1, 7)
                boxes[:, 6] = limit_period(boxes[:, 6], 0.5, 2 * np.pi)
                gt_boxes.append(boxes)
        # cloud_sizes = [cloud.shape[0] for cloud in clouds]
        # clouds = np.concatenate(clouds, axis=0)
        cloud_fused = np.concatenate(cloud_fused, axis=0)
        translations = np.stack(translations, axis=0)
        coop_boxes_in_egoCS = gt_boxes[0][[int(np.where((boxes[:, :2]==0).all(axis=1))[0])
                                           for boxes in gt_boxes]]

        # clouds = clouds[self._mask_points_in_box(clouds[:, :3], self.cfg.pc_range)]
        # from vlib.visulization import draw_points_boxes_plt
        # draw_points_boxes_plt(self.cfg.pc_range, clouds, None, gt_boxes, False)
        # print('debug')
        # gt_boxes[:, 3:6] = gt_boxes[:, 3:6] / 2
        batch_type = {
            "ids": "cpu_none",
            "points": "gpu_float",
            "translations": "gpu_float",
            "gt_boxes": "gpu_float",
            "coop_boxes_in_egoCS": "gpu_float",
            "frame": "cpu_none"
        }

        return {
            "ids": ids,
            "points": clouds,
            "cloud_fused": cloud_fused,
            "translations": translations,
            "gt_boxes": gt_boxes,
            "coop_boxes_in_egoCS": coop_boxes_in_egoCS,
            "frame": self.file_list[index],
            "batch_types": batch_type
        }

    def mask_points_in_range(self, data_dict):
        # mask points and boxes outside range
        clouds = data_dict["points"]
        gt_boxes = data_dict["gt_boxes"]
        clouds_out = []
        gt_boxes_out = []
        for cloud, boxes in zip(clouds, gt_boxes):
            box_centers = boxes[:, :3].astype(float)
            if self.cfg.range_clip_mode=='circle':
                mask_points = self._mask_points_in_range(cloud[:, :3], self.cfg.pc_range[3])
                mask_boxes = self._mask_points_in_range(box_centers, self.cfg.pc_range[3])
            elif self.cfg.range_clip_mode=='rectangle':
                mask_points = self._mask_points_in_box(cloud[:, :3], self.cfg.pc_range)
                mask_boxes = self._mask_points_in_box(box_centers, self.cfg.pc_range)
            else:
                raise NotImplementedError
            clouds_out.append(cloud[mask_points])
            gt_boxes_out.append(boxes[mask_boxes])
        data_dict["points"] = clouds_out
        data_dict["batch_types"]["points_labels"] = "gpu_long"
        data_dict["gt_boxes"] = gt_boxes_out
        data_dict["gt_boxes_fused"] = gt_boxes_out[0]
        data_dict["batch_types"]["gt_boxes_fused"] = "gpu_float"
        data_dict.update({
            "gt_names": [np.array(["Car"] * len(gt_boxes_out[i])) for i in range(len(gt_boxes_out))],
            "gt_classes": [np.array([1] * len(gt_boxes_out[i])) for i in range(len(gt_boxes_out))],
            "gt_names_fused": np.array(["Car"] * len(data_dict["gt_boxes_fused"])),
            "gt_classes_fused": np.array([1] * len(data_dict["gt_boxes_fused"])),
        })
        return data_dict

    def rm_empty_gt_boxes(self, data_dict):
        clouds = data_dict["points"]
        gt_boxes_list = data_dict["gt_boxes"]
        boxes_out = []
        for cloud, gt_boxes in zip(clouds, gt_boxes_list):
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(cloud, gt_boxes)
            selected = point_indices.sum(axis=1) > 5
            boxes_out.append(gt_boxes[selected])
        data_dict["gt_boxes"] = boxes_out
        # remove empty boxes for final detection
        shifts = data_dict["translations"] - data_dict["translations"][0:1, :]
        clouds_fused = []
        for cloud, shift in zip(clouds, shifts):
            clouds_fused.append(cloud + shift.reshape(-1, 3))
        cloud_fused = np.concatenate(clouds_fused, axis=0)
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(cloud_fused, gt_boxes_list[0])
        selected = point_indices.sum(axis=1) > 5
        data_dict["gt_boxes_fused"] = gt_boxes_list[0][selected]
        data_dict.update({
            "gt_names": [np.array(["Car"] * len(boxes_out[i])) for i in range(len(boxes_out))],
            "gt_classes": [np.array([1] * len(boxes_out[i])) for i in range(len(boxes_out))],
            "gt_names_fused": np.array(["Car"] * len(data_dict["gt_boxes_fused"])),
            "gt_classes_fused": np.array([1] * len(data_dict["gt_boxes_fused"])),
        })
        return data_dict

    def points_to_voxel(self, data_dict):
        clouds = data_dict["points"]
        voxel_output = {"voxels": [], "coordinates": [], "num_points_per_voxel": []}
        for cloud in clouds:
            cur_voxel_output = self.voxel_generator.generate(cloud[:, :3])
            voxel_output["voxels"].append(cur_voxel_output["voxels"])
            voxel_output["coordinates"].append(cur_voxel_output["coordinates"])
            voxel_output["num_points_per_voxel"].append(cur_voxel_output["num_points_per_voxel"])
        # generate voxels for raw fused cloud
        voxel_output_fused = self.voxel_generator.generate(data_dict["cloud_fused"])
        data_dict.update({
            "voxels": voxel_output["voxels"],
            "voxel_coords": voxel_output["coordinates"],
            "voxel_num_points": voxel_output["num_points_per_voxel"],
            "fused_voxels": [voxel_output_fused["voxels"]],
            "fused_voxel_coords": [voxel_output_fused["coordinates"]],
            "fused_voxel_num_points": [voxel_output_fused["num_points_per_voxel"]]
        })
        data_dict["batch_types"].update({
            "voxels": "gpu_float",
            "voxel_coords": "gpu_float",
            "voxel_num_points": "gpu_float",
            "fused_voxels": "gpu_float",
            "fused_voxel_coords": "gpu_float",
            "fused_voxel_num_points": "gpu_float"
        })

        return data_dict
    
    def assign_target(self, data_dict):
        assert self.target_assigner is not None
        # assign target for single agent detection
        if self.training:
            targets = {"labels": [], "reg_targets": [], "reg_weights": [],
                       "positive_gt_id": [], "anchors": []}
        else:
            targets = {"anchors": []}
        for i in range(len(data_dict['points'])):
            data_dict_i = {
                "gt_boxes": data_dict["gt_boxes"][i],
                "gt_names": data_dict["gt_names"][i],
                "gt_classes": data_dict["gt_classes"][i],
                "batch_types": data_dict["batch_types"],
            }
            data_dict_i = self.target_assigner(data_dict_i)
            if self.training:
                targets["labels"].append(data_dict_i["labels"])
                targets["reg_targets"].append(data_dict_i["reg_targets"])
                targets["reg_weights"].append(data_dict_i["reg_weights"])
                targets["positive_gt_id"].append(data_dict_i["positive_gt_id"])
            targets["anchors"].append(data_dict_i["anchors"])
        data_dict.update(targets)
        # assign target for multi agent fused detection
        data_dict_fused = {
            "gt_boxes": data_dict["gt_boxes_fused"],
            "gt_names": data_dict.pop("gt_names_fused"),
            "gt_classes": data_dict.pop("gt_classes_fused"),
            "batch_types": data_dict["batch_types"],
        }
        data_dict_fused = self.target_assigner(data_dict_fused)
        if self.training:
            data_dict["target_fused"] = {
                "labels"        : data_dict_fused["labels"],
                "reg_targets"   : data_dict_fused["reg_targets"],
                "reg_weights"   : data_dict_fused["reg_weights"],
                "positive_gt_id": data_dict_fused["positive_gt_id"]
            }
        
        return data_dict

    def drop_intermidiate_data(self, data_dict):
        data_dict.pop('gt_names')
        data_dict.pop('gt_classes')
        return data_dict

    def _mask_values_in_range(self, values, min,  max):
        return np.logical_and(values>min, values<max)

    def _mask_points_in_box(self, points, pc_range):
        n_ranges = len(pc_range) // 2
        list_mask = [self._mask_values_in_range(points[:,i], pc_range[i],
                                                pc_range[i+n_ranges]) for i in range(n_ranges)]
        return np.array(list_mask).all(axis=0)

    def _mask_points_in_range(self, points, dist):
        return np.linalg.norm(points[:, :2], axis=1) < dist

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        batch_types = [data.pop("batch_types") for data in batch_list][0]
        batch_size = len(batch_list)
        assert batch_size==1, "Only support batch size 1."
        data_dict = batch_list[0]

        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ["voxels", "voxel_num_points", "fused_voxels", "fused_voxel_num_points"]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ["points",  "voxel_coords",  "fused_voxel_coords"]:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ["frame", "gt_boxes", "positive_gt_id", "translations", "ids",
                             "target_fused", "coop_boxes_in_egoCS"]:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print("Error in collate_batch: key=%s" % key)
                raise TypeError


        ret["batch_size"] = len(data_dict['points'])
        batch_types["batch_size"] = "cpu_none"
        ret["batch_types"] = batch_types
        return ret


    @staticmethod
    def _collate_batch(batch_list, _unused=False):
        batch_types = [data.pop("batch_types") for data in batch_list][0]
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)

        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ["voxels", "voxel_num_points"]:
                    ret[key] = [np.concatenate(val[b], axis=0) for b in range(batch_size)]
                elif key in ["voxel_coords"]:
                    coors_list = []
                    for b, cur_val in enumerate(val):
                        coors = []
                        for i, coor in enumerate(cur_val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                            coors.append(coor_pad)
                        coors_list.append(np.concatenate(coors, axis=0))
                    ret[key] = coors_list
                elif key in ["points_labels"]:
                    max_gt = max([len(x) for x in val])
                    assert len(val[0].shape)==2
                    batch_points = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_points[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_points
                elif key in ["encoded_gt_boxes"]:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ["frame", "gt_boxes", "positive_gt_id", "points", "translations"]:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print("Error in collate_batch: key=%s" % key)
                raise TypeError

        ret["batch_size"] = batch_size
        batch_types["batch_size"] = "cpu_none"
        ret["batch_types"] = batch_types
        return ret


def split(cfg):
    path_ego = Path(cfg.root) / "cloud_ego"
    list_train_val = []
    list_test = []
    for filename in path_ego.glob("*.bin"):
        if filename.name.split("_")[0] in cfg.train_val_split:
            list_train_val.append(filename.name[:-4])
        elif filename.name.split("_")[0] in cfg.test_split:
            list_test.append(filename.name[:-4])

    with open(cfg.root + "/train_val.txt", "w") as fa:
        for line in list_train_val:
            fa.writelines(line + "\n")
    with open(cfg.root + "/test.txt", "w") as fb:
        for line in list_test:
            fb.writelines(line + "\n")


if __name__=="__main__":
    from cfg.fusion_ssd_comap import Dataset
    dcfg = Dataset()
    train_dataset = CoMapDataset(dcfg, training=True)
    sampler = None
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, pin_memory=True, num_workers=1,
        shuffle=True, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )
    for batch_data in train_dataloader:
        pass



