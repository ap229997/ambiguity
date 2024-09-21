import os
import os.path as op
import json
import pickle
import cv2

import numpy as np
from numpy.lib.index_tricks import r_
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d


class CropDataset(Dataset):
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        imgname = imgname.replace("./", os.environ["DATA_DIR"] + "/")
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE:
        # kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        # objects
        bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
        bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
        bbox2d_t = bbox2d[:8]
        bbox2d_b = bbox2d[8:]
        bbox3d_t = bbox3d[:8]
        bbox3d_b = bbox3d[8:]

        kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
        kp3d = data_cam["kp3d"][vidx, view_idx].copy()
        kp2d_t = kp2d[:16]
        kp2d_b = kp2d[16:]
        kp3d_t = kp3d[:16]
        kp3d_b = kp3d[16:]

        obj_radian = data_params["obj_arti"][vidx].copy()

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        # LOADING END

        # SPEEDUP PROCESS
        (
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup(
            speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "/arctic_data/", "/data/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        kp2d_b = data_utils.j2d_processing(
            kp2d_b, center, scale, augm_dict, args.img_res
        )
        kp2d_t = data_utils.j2d_processing(
            kp2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d_b = data_utils.j2d_processing(
            bbox2d_b, center, scale, augm_dict, args.img_res
        )
        bbox2d_t = data_utils.j2d_processing(
            bbox2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d = np.concatenate((bbox2d_t, bbox2d_b), axis=0)
        kp2d = np.concatenate((kp2d_t, kp2d_b), axis=0)

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )

            # bbox format below: [x0, y0, w, h] for right_bbox, left_bbox
            if 'train' in self.split or args.use_gt_bbox:
                # compute bbox from ground truth joints2d during training
                j2d_r_pix = ((joints2d_r[...,:2]+1)/2)*(args.img_res-1)
                j2d_l_pix = ((joints2d_l[...,:2]+1)/2)*(args.img_res-1)
                right_bbox = np.array([j2d_r_pix[...,0].min(), j2d_r_pix[...,1].min(), j2d_r_pix[...,0].max(), j2d_r_pix[...,1].max()]).clip(0, args.img_res-1)
                left_bbox = np.array([j2d_l_pix[...,0].min(), j2d_l_pix[...,1].min(), j2d_l_pix[...,0].max(), j2d_l_pix[...,1].max()]).clip(0, args.img_res-1)
                right_bbox = np.array([right_bbox[0], right_bbox[1], right_bbox[2]-right_bbox[0], right_bbox[3]-right_bbox[1]]).astype(np.int16)
                left_bbox = np.array([left_bbox[0], left_bbox[1], left_bbox[2]-left_bbox[0], left_bbox[3]-left_bbox[1]]).astype(np.int16)
                if right_bbox[2] == 0 or right_bbox[3] == 0: right_bbox = None # no right hand in the image
                if left_bbox[2] == 0 or left_bbox[3] == 0: left_bbox = None # no left hand in the image

                if args.get('use_obj_bbox', False):
                    bbox2d_pix = ((bbox2d[...,:2]+1)/2)*(args.img_res-1)
                    obj_bbox = np.array([bbox2d_pix[...,0].min(), bbox2d_pix[...,1].min(), bbox2d_pix[...,0].max(), bbox2d_pix[...,1].max()]).clip(0, args.img_res-1)
                    obj_bbox = np.array([obj_bbox[0], obj_bbox[1], obj_bbox[2]-obj_bbox[0], obj_bbox[3]-obj_bbox[1]]).astype(np.int16)
                    if obj_bbox[2] == 0 or obj_bbox[3] == 0: obj_bbox = None # no object in the image
            else:
                # get bounding box information from pretrained bbox detectors (frankmocap or arctic models)
                hand_bbox = self.bbox_dict[imgname]
                right_bbox, left_bbox = hand_bbox['bbox_right'], hand_bbox['bbox_left']
                if args.get('use_obj_bbox', False):
                    obj_bbox = hand_bbox['bbox_obj']
            
            if self.aug_data:
                right_bbox, left_bbox = data_utils.jitter_bbox(right_bbox), data_utils.jitter_bbox(left_bbox)
                if right_bbox is not None:
                    new_right_bbox = np.array([right_bbox[0], right_bbox[1], right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                    if (new_right_bbox[2]-new_right_bbox[0]) == 0 or (new_right_bbox[3]-new_right_bbox[1]) == 0: right_bbox = None
                if left_bbox is not None:
                    new_left_bbox = np.array([left_bbox[0], left_bbox[1], left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                    if (new_left_bbox[2]-new_left_bbox[0]) == 0 or (new_left_bbox[3]-new_left_bbox[1]) == 0: left_bbox = None
                # augmention for object bbox
                if args.get('use_obj_bbox', False) and obj_bbox is not None:
                    obj_bbox = data_utils.jitter_bbox(obj_bbox)
                    new_obj_bbox = np.array([obj_bbox[0], obj_bbox[1], obj_bbox[0]+obj_bbox[2], obj_bbox[1]+obj_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                    if (new_obj_bbox[2]-new_obj_bbox[0]) == 0 or (new_obj_bbox[3]-new_obj_bbox[1]) == 0: obj_bbox = None
            
            # bbox format below; [x0, y0, x1, y1] for r_bbox, l_bbox
            r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, args, scale=1.5)
            l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, args, scale=1.5)
            norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
            norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

            if args.get('use_obj_bbox', False):
                obj_img, obj_bbox = data_utils.crop_and_pad(img, obj_bbox, args, scale=1.5)
                norm_obj_img = self.normalize_img(torch.from_numpy(obj_img).float())

            img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
            img_ds = np.clip(img_ds, 0, 1)
            img_ds = torch.from_numpy(img_ds).float()
            norm_img = self.normalize_img(img_ds)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox

        if args.get('use_obj_bbox', False):
            inputs["obj_img"] = norm_obj_img
            inputs["obj_bbox"] = obj_bbox
            obj_mask = np.zeros((args.img_res_ds, args.img_res_ds))
            if obj_bbox is not None:
                obj_mask[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2]] = 1

        meta_info["imgname"] = imgname
        rot_r = data_cam["rot_r_cam"][vidx, view_idx]
        rot_l = data_cam["rot_l_cam"][vidx, view_idx]

        pose_r = np.concatenate((rot_r, pose_r), axis=0)
        pose_l = np.concatenate((rot_l, pose_l), axis=0)

        # hands
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            data_utils.pose_processing(pose_l, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # object
        targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()
        targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()
        targets["object.kp3d.full.t"] = torch.from_numpy(kp3d_t[:, :3]).float()
        targets["object.kp2d.norm.t"] = torch.from_numpy(kp2d_t[:, :2]).float()

        targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()
        targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()
        targets["object.bbox3d.full.t"] = torch.from_numpy(bbox3d_t[:, :3]).float()
        targets["object.bbox2d.norm.t"] = torch.from_numpy(bbox2d_t[:, :2]).float()
        targets["object.radian"] = torch.FloatTensor(np.array(obj_radian))

        targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
        targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

        # compute RT from cano space to augmented space
        # this transform match j3d processing
        obj_idx = self.obj_names.index(obj_name)
        meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000  # meter
        kp3d_cano = meta_info["kp3d.cano"].numpy()
        kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

        # rotate canonical kp3d to match original image
        R, _ = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        obj_rot = (
            rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        )

        # multiply rotation from data augmentation
        obj_rot_aug = rot.rot_aa(obj_rot, augm_dict["rot"])
        targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)

        # full image camera coord
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])
        targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])
        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k
        else:
            intrx = intrx.numpy() # make format consistent with gt intrinsics used in egocentric setting
 
        if args.pos_enc is not None: # compute positional encoding for different pixels
            L = args.n_freq_pos_enc
            
            if args.pos_enc is not None and ('center' in args.pos_enc or 'corner' in args.pos_enc):
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(r_center[1]-intrx[1,2], intrx[1,1])
                l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(l_center[1]-intrx[1,2], intrx[1,1])
                r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
                l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle
                
                if args.get('use_obj_bbox', False):
                    obj_center = (obj_bbox[:2] + obj_bbox[2:]) / 2.0
                    obj_angle_x, obj_angle_y = np.arctan2(obj_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(obj_center[1]-intrx[1,2], intrx[1,1])
                    obj_angle = np.array([obj_angle_x, obj_angle_y]).astype(np.float32)
                    inputs["obj_center_angle"] = obj_angle
                    targets['center.obj'] = obj_angle

                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_corner = np.stack([r_corner[:,0]-intrx[0,2], r_corner[:,1]-intrx[1,2]], axis=-1)
                l_corner = np.stack([l_corner[:,0]-intrx[0,2], l_corner[:,1]-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                l_angle = np.arctan2(l_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle
                
                if args.get('use_obj_bbox', False):
                    obj_corner = np.array([[obj_bbox[0], obj_bbox[1]], [obj_bbox[0], obj_bbox[3]], [obj_bbox[2], obj_bbox[1]], [obj_bbox[2], obj_bbox[3]]])
                    obj_corner = np.stack([obj_corner[:,0]-intrx[0,2], obj_corner[:,1]-intrx[1,2]], axis=-1)
                    obj_angle = np.arctan2(obj_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                    inputs["obj_corner_angle"] = obj_angle
                    targets['corner.obj'] = obj_angle

            if args.pos_enc is not None and 'dense' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx[0,2], r_y_grid-intrx[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx[0,2], l_y_grid-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)

                if args.get('use_obj_bbox', False):
                    obj_x_grid, obj_y_grid = range(obj_bbox[0], obj_bbox[2]+1), range(obj_bbox[1], obj_bbox[3]+1)
                    obj_x_grid, obj_y_grid = np.meshgrid(obj_x_grid, obj_y_grid, indexing='ij')
                    obj_pix = np.stack([obj_x_grid-intrx[0,2], obj_y_grid-intrx[1,2]], axis=-1)
                    obj_angle = np.arctan2(obj_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)

                r_angle_fdim = np.zeros((r_angle.shape[0], args.img_res, args.img_res))
                l_angle_fdim = np.zeros((l_angle.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)
                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)

                if args.get('use_obj_bbox', False):
                    obj_angle_fdim = np.zeros((obj_angle.shape[0], args.img_res, args.img_res))
                    obj_angle_fdim[:obj_angle.shape[0], :obj_angle.shape[1], :obj_angle.shape[2]] = obj_angle
                    inputs["obj_dense_angle"] = obj_angle_fdim.astype(np.float32)
                    obj_mask = np.zeros((args.img_res, args.img_res))
                    obj_mask[:obj_angle.shape[1], :obj_angle.shape[2]] = 1
                    inputs["obj_dense_mask"] = obj_mask.astype(np.float32)

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        # meta_info["sample_index"] = index

        # root and at least 3 joints inside image
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * float(is_valid)
        targets["right_valid"] = float(right_valid) * float(is_valid)
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames

    def _load_data(self, args, split, seq):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"{os.environ['DATA_DIR']}/data/arctic_data/data/splits/{args.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        with open(f"{os.environ['DATA_DIR']}/data/arctic_data/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        if not args.use_gt_bbox and ('val' in self.split or 'test' in self.split): # only during validation or test
            if args.setup == 'p1':
                raise NotImplementedError
            bbox_file = "weights/bbox_detector/bbox_p2_val_obj.pkl"
            with open(bbox_file, "rb") as f:
                bbox_dict = pickle.load(f)
            self.bbox_dict = bbox_dict

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

    def __init__(self, args, split, seq=None):
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        if self.args.debug:
            return 2
        return len(self.imgnames)

    def getitem_eval(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        
        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE:
        # kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        # LOADING END

        # SPEEDUP PROCESS
        bbox = dataset_utils.transform_bbox_for_speedup(
            speedup,
            is_egocam,
            bbox,
            args.ego_image_scale,
        )

        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "/arctic_data/", "/data/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )

            hand_bbox = self.bbox_dict[imgname]
            right_bbox, left_bbox = hand_bbox['bbox_right'], hand_bbox['bbox_left']
            if self.args.get('use_obj_bbox', False):
                obj_bbox = hand_bbox['bbox_obj']
            
            # bbox format below; [x0, y0, x1, y1] for r_bbox, l_bbox
            r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, args, scale=1.5)
            l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, args, scale=1.5)
            norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
            norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

            if args.get('use_obj_bbox', False):
                obj_img, obj_bbox = data_utils.data_utils.crop_and_pad(img, obj_bbox, args, scale=1.5)
                norm_obj_img = self.normalize_img(torch.from_numpy(obj_img).float())

            img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
            img_ds = np.clip(img_ds, 0, 1)
            img_ds = torch.from_numpy(img_ds).float()
            norm_img = self.normalize_img(img_ds)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox

        if args.get('use_obj_bbox', False):
            inputs["obj_img"] = norm_obj_img
            inputs["obj_bbox"] = obj_bbox

        meta_info["imgname"] = imgname

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k
        else:
            intrx = intrx.numpy() # make format consistent with gt intrinsics used in egocentric setting
        
        if args.pos_enc is not None: # compute positional encoding for different pixels
            L = args.n_freq_pos_enc
            
            if args.pos_enc is not None and ('center' in args.pos_enc or 'corner' in args.pos_enc):
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(r_center[1]-intrx[1,2], intrx[1,1])
                l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(l_center[1]-intrx[1,2], intrx[1,1])
                r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
                l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle
                
                if args.get('use_obj_bbox', False):
                    obj_center = (obj_bbox[:2] + obj_bbox[2:]) / 2.0
                    obj_angle_x, obj_angle_y = np.arctan2(obj_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(obj_center[1]-intrx[1,2], intrx[1,1])
                    obj_angle = np.array([obj_angle_x, obj_angle_y]).astype(np.float32)
                    inputs["obj_center_angle"] = obj_angle
                    targets['center.obj'] = obj_angle

                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_corner = np.stack([r_corner[:,0]-intrx[0,2], r_corner[:,1]-intrx[1,2]], axis=-1)
                l_corner = np.stack([l_corner[:,0]-intrx[0,2], l_corner[:,1]-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                l_angle = np.arctan2(l_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle
                
                if args.get('use_obj_bbox', False):
                    obj_corner = np.array([[obj_bbox[0], obj_bbox[1]], [obj_bbox[0], obj_bbox[3]], [obj_bbox[2], obj_bbox[1]], [obj_bbox[2], obj_bbox[3]]])
                    obj_corner = np.stack([obj_corner[:,0]-intrx[0,2], obj_corner[:,1]-intrx[1,2]], axis=-1)
                    obj_angle = np.arctan2(obj_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                    inputs["obj_corner_angle"] = obj_angle
                    targets['corner.obj'] = obj_angle

            if args.pos_enc is not None and 'dense' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx[0,2], r_y_grid-intrx[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx[0,2], l_y_grid-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)

                if args.get('use_obj_bbox', False):
                    obj_x_grid, obj_y_grid = range(obj_bbox[0], obj_bbox[2]+1), range(obj_bbox[1], obj_bbox[3]+1)
                    obj_x_grid, obj_y_grid = np.meshgrid(obj_x_grid, obj_y_grid, indexing='ij')
                    obj_pix = np.stack([obj_x_grid-intrx[0,2], obj_y_grid-intrx[1,2]], axis=-1)
                    obj_angle = np.arctan2(obj_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)

                r_angle_fdim = np.zeros((r_angle.shape[0], args.img_res, args.img_res))
                l_angle_fdim = np.zeros((l_angle.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)
                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)

                if args.get('use_obj_bbox', False):
                    obj_angle_fdim = np.zeros((obj_angle.shape[0], args.img_res, args.img_res))
                    obj_angle_fdim[:obj_angle.shape[0], :obj_angle.shape[1], :obj_angle.shape[2]] = obj_angle
                    inputs["obj_dense_angle"] = obj_angle_fdim.astype(np.float32)
                    obj_mask = np.zeros((args.img_res, args.img_res))
                    obj_mask[:obj_angle.shape[1], :obj_angle.shape[2]] = 1
                    inputs["obj_dense_mask"] = obj_mask.astype(np.float32)

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])

        return inputs, targets, meta_info