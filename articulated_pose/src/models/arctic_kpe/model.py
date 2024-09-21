import torch
import torch.nn as nn
import torch.nn.functional as F

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


class ArcticKPE(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(ArcticKPE, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
        else:
            assert False
        self.backbone = resnet(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.feat_dim = feat_dim
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)

        self.hand_backbone = resnet(pretrained=True)
        conv1 = self.hand_backbone.conv1

        if args.get('use_obj_bbox', False):
            self.obj_backbone = resnet(pretrained=True)
            conv1_obj = self.obj_backbone.conv1
        
        if args.pos_enc == 'center': inp_dim = 3 + 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'corner': inp_dim = 3 + 4 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'center+corner': inp_dim = 3 + 5 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'dense': inp_dim = 3 + 4 * args.n_freq_pos_enc
        else: inp_dim = 3
        
        if inp_dim != conv1.in_channels:
            self.hand_backbone.conv1 = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)

        if args.get('use_obj_bbox', False) and inp_dim != conv1_obj.in_channels:
            self.obj_backbone.conv1 = nn.Conv2d(inp_dim, conv1_obj.out_channels, kernel_size=conv1_obj.kernel_size, stride=conv1_obj.stride, padding=conv1_obj.padding, bias=conv1_obj.bias)

        if args.pos_enc == 'dense_latent':
            feat_conv_dim = feat_dim + 4 * args.n_freq_pos_enc
            pos_enc_dim = 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'center+corner_latent':
            feat_conv_dim = feat_dim + 5 * 4 * args.n_freq_pos_enc
            pos_enc_dim = 5 * 4 * args.n_freq_pos_enc
        else:
            feat_conv_dim = feat_dim
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(feat_conv_dim, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, feat_dim),
            nn.ReLU(inplace=True),
        )

        if args.get('use_obj_bbox', False):
            self.obj_bbox_feature_conv = nn.Sequential(
                nn.Conv2d(feat_conv_dim, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(256 * 3 * 3, feat_dim),
                nn.ReLU(inplace=True),
            )
        
        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]

        features = self.backbone(images)
        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        bz, c, w, h = inputs["r_img"].shape
        r_inp = inputs["r_img"]
        l_inp = inputs["l_img"]
        
        r_features = self.hand_backbone(r_inp)
        l_features = self.hand_backbone(l_inp)

        if self.args.pos_enc == 'dense_latent':
            r_dense_pos_enc = self.compute_dense_pos_enc(inputs["r_dense_angle"], inputs["r_dense_mask"])
            l_dense_pos_enc = self.compute_dense_pos_enc(inputs["l_dense_angle"], inputs["l_dense_mask"])
            c, w, h = r_features.shape[1:]
            r_dense_pos_enc = F.interpolate(r_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
            l_dense_pos_enc = F.interpolate(l_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
            
            r_features = torch.cat([r_features+features, r_dense_pos_enc], dim=1)
            l_features = torch.cat([l_features+features, l_dense_pos_enc], dim=1)

            hand_enc = torch.cat([r_dense_pos_enc, l_dense_pos_enc], dim=1) # to be used for object prediction
        
        elif self.args.pos_enc == 'center+corner_latent':
            r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
            l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
            r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
            l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
            
            c, w, h = r_features.shape[1:]
            r_center_pos_enc = r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
            l_center_pos_enc = l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
            r_corner_pos_enc = r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
            l_corner_pos_enc = l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)

            r_features = torch.cat([r_features+features, r_center_pos_enc, r_corner_pos_enc], dim=1)
            l_features = torch.cat([l_features+features, l_center_pos_enc, l_corner_pos_enc], dim=1)
            hand_enc = torch.cat([r_center_pos_enc, r_corner_pos_enc, l_center_pos_enc, l_corner_pos_enc], dim=1) # to be used for object prediction

        r_features = self.feature_conv(r_features)
        l_features = self.feature_conv(l_features)

        if self.args.get('use_obj_bbox', False):
            obj_inp = inputs["obj_img"]
            obj_features = self.obj_backbone(obj_inp)
            
            if self.args.pos_enc == 'dense_latent':
                obj_dense_pos_enc = self.compute_dense_pos_enc(inputs["obj_dense_angle"], inputs["obj_dense_mask"])
                c_obj, w_obj, h_obj = obj_features.shape[1:]
                obj_dense_pos_enc = F.interpolate(obj_dense_pos_enc, size=(w_obj,h_obj), mode='bilinear', align_corners=True)
                obj_features = torch.cat([obj_features+features, obj_dense_pos_enc], dim=1)
            
            elif self.args.pos_enc == 'center+corner_latent':
                obj_center_pos_enc = self.compute_center_pos_enc(inputs["obj_center_angle"])
                obj_corner_pos_enc = self.compute_corner_pos_enc(inputs["obj_corner_angle"])
                
                c_obj, w_obj, h_obj = obj_features.shape[1:]
                obj_center_pos_enc = obj_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w_obj,h_obj)
                obj_corner_pos_enc = obj_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w_obj,h_obj)
                
                obj_features = torch.cat([obj_features+features, obj_center_pos_enc, obj_corner_pos_enc], dim=1)

            obj_features = self.obj_bbox_feature_conv(obj_features)

        hmr_output_r = self.head_r(r_features, use_pool=False)
        hmr_output_l = self.head_l(l_features, use_pool=False)

        if self.args.get('use_obj_bbox', False):
            hmr_output_o = self.head_o(obj_features, use_pool=False)
        else:
            hmr_output_o = self.head_o(features)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_o["rot"],
            angle=hmr_output_o["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]
        root_o_init = hmr_output_o["cam_t.wp.init"]
        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init
        arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()

        return output

    def compute_center_pos_enc(self, angle):
        # center positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        center_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return center_pos_enc

    def compute_corner_pos_enc(self, angle):
        # corner positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        corner_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return corner_pos_enc

    def compute_dense_pos_enc(self, angle, mask):
        # dense positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c, w, h = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1,1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c,w,h)
        dense_pos_enc = torch.cat([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=3).reshape(bz, -1, w, h).float()
        mask_expand = mask.unsqueeze(1).repeat(1,2*L*c,1,1)
        dense_pos_enc = dense_pos_enc * mask_expand
        dense_pos_enc = F.interpolate(dense_pos_enc, size=(self.args.img_res_ds, self.args.img_res_ds), mode='bilinear', align_corners=True)
        return dense_pos_enc