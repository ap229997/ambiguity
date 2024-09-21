# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import itertools

import torch
import torch.nn as nn
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)
from zoedepth.models.model_io import load_state_from_resource


class ZoeDepth(DepthModel):
    def __init__(self, core,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv
        
        img_size = kwargs.get('img_size', '384,512')
        if isinstance(img_size, str):
            self.img_size = [int(x) for x in img_size.split(',')]
        else:
            self.img_size = img_size
        
        self.pos_enc = kwargs.get('pos_enc', None)
        if self.pos_enc is not None:
            self.n_freq_pos_enc = kwargs.get('n_freq_pos_enc') # should always be present for pos_enc
            feat_conv_dim = btlnck_features
            if self.pos_enc == 'center+corner_latent':
                pos_enc_dim = 5 * 4 * self.n_freq_pos_enc
                feat_conv_dim = btlnck_features + pos_enc_dim
            elif self.pos_enc == 'dense_latent':
                pos_enc_dim = 4 * self.n_freq_pos_enc
                feat_conv_dim = btlnck_features + pos_enc_dim

            self.midas_pos_enc = kwargs.get('midas_pos_enc', False)
            if self.midas_pos_enc:
                self.downsample_factor = self.core.core.pretrained.model.patch_size[0]
                self.midas_tf_feat_dim = self.core.core.pretrained.model.patch_embed.proj.out_channels
                self.conv_pos_enc = nn.Sequential(
                    nn.Conv2d(pos_enc_dim, 256, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.midas_tf_feat_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )
            else:
                self.feature_conv = nn.Sequential(
                    nn.Conv2d(feat_conv_dim, btlnck_features, kernel_size=1, stride=1, padding=0, bias=(not kwargs.get('no_bias', False))), # use bias
                    nn.ReLU(inplace=True),
                    nn.Conv2d(btlnck_features, btlnck_features, kernel_size=1, stride=1, padding=0, bias=(not kwargs.get('no_bias', False))),
                    nn.ReLU(inplace=True),
                )

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        b, c, h, w = x.shape
        # print("input shape ", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        
        more_args = {}
        if self.pos_enc is not None:
            if self.pos_enc == 'dense_latent':
                dense_angle, dense_mask = kwargs.get('dense_angle'), kwargs.get('dense_mask') # should be present
                dense_angle, dense_mask = dense_angle.permute(0,1,3,2), dense_mask.permute(0,2,1)
                dense_pos_enc = self.compute_dense_pos_enc(dense_angle, dense_mask, size=(h,w))
                if self.midas_pos_enc:
                    tf_h, tf_w = int(self.img_size[0]/self.downsample_factor), int(self.img_size[1]/self.downsample_factor)
                    tf_pos_enc = torch.nn.functional.interpolate(dense_pos_enc, size=(tf_h,tf_w), mode='bilinear', align_corners=True)
                    tf_pos_enc = self.conv_pos_enc(tf_pos_enc).permute(0,2,3,1).view(b,-1,self.midas_tf_feat_dim)
                    more_args['tf_pos_enc'] = tf_pos_enc

            elif self.pos_enc == 'center+corner_latent':
                center_angle, corner_angle = kwargs.get('center_angle'), kwargs.get('corner_angle') # should be present
                center_pos_enc = self.compute_center_pos_enc(center_angle)
                corner_pos_enc = self.compute_corner_pos_enc(corner_angle)
                more_args['center_pos_enc'] = center_pos_enc
                more_args['corner_pos_enc'] = corner_pos_enc
                if self.midas_pos_enc:
                    tf_h, tf_w = int(self.img_size[0]/self.downsample_factor), int(self.img_size[1]/self.downsample_factor)
                    tf_center_pos_enc = center_pos_enc.view(b,-1,1,1).repeat(1,1,tf_h,tf_w)
                    tf_corner_pos_enc = corner_pos_enc.view(b,-1,1,1).repeat(1,1,tf_h,tf_w)
                    tf_pos_enc = torch.cat([tf_center_pos_enc, tf_corner_pos_enc], dim=1)
                    tf_pos_enc = self.conv_pos_enc(tf_pos_enc).permute(0,2,3,1).view(b,-1,self.midas_tf_feat_dim)
                    more_args['tf_pos_enc'] = tf_pos_enc

        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True, **more_args)
        # print("output shapes", rel_depth.shape, out.shape)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        if self.pos_enc is not None and not self.midas_pos_enc:
            if self.pos_enc == 'dense_latent':
                fh, fw = btlnck.shape[2:]
                ang_pos_enc = torch.nn.functional.interpolate(dense_pos_enc, size=(fh,fw), mode='bilinear', align_corners=True)

            elif self.pos_enc == 'center+corner_latent':
                bz, _, fh, fw = btlnck.shape
                center_pos_enc = center_pos_enc.view(bz,-1,1,1).repeat(1,1,fh,fw)
                corner_pos_enc = corner_pos_enc.view(bz,-1,1,1).repeat(1,1,fh,fw)
                ang_pos_enc = torch.cat([center_pos_enc, corner_pos_enc], dim=1)
                
            btlnck = torch.cat([btlnck, ang_pos_enc], dim=1)
            btlnck = self.feature_conv(btlnck)

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        metric_depth = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=metric_depth)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x
        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            midas_params = self.core.core.scratch.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth.build(**config)

    def compute_center_pos_enc(self, angle):
        # center positional encoding for all pixels
        L = self.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        center_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return center_pos_enc

    def compute_corner_pos_enc(self, angle):
        # corner positional encoding for all pixels
        L = self.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        corner_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return corner_pos_enc

    def compute_dense_pos_enc(self, angle, mask, size):
        # dense positional encoding for all pixels
        L = self.n_freq_pos_enc
        bz, c, w, h = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1,1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c,w,h)
        dense_pos_enc = torch.cat([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=3).reshape(bz, -1, w, h).float()
        mask_expand = mask.unsqueeze(1).repeat(1,2*L*c,1,1)
        dense_pos_enc = dense_pos_enc * mask_expand
        dense_pos_enc = torch.nn.functional.interpolate(dense_pos_enc, size=size, mode='bilinear', align_corners=True)
        return dense_pos_enc
