# BEV-PROJECT, MODIFIED

# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone

__all__ = ["build_resnet_fpn_backbone", "build_resnet_sparse_mask_fpn_backbone", "build_resnet_sparse_batch_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "FPN", "SparseMiniFPN"]


class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        print("SIZE DIV:", self._size_divisibility)
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_sparse_mask_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


# BEV-project, CREATED

def block_sparse_if_all(sparse_mask : torch.Tensor, block_size : int):
    assert block_size == sparse_mask.shape[-1]
    assert block_size == sparse_mask.shape[-2]
    return torch.any(sparse_mask.view(*sparse_mask.shape[:-2], -1), dim=-1)


def block_sparse_if_count(sparse_mask : torch.Tensor, block_size : int, count : int):
    assert block_size == sparse_mask.shape[-1]
    assert block_size == sparse_mask.shape[-2]
    a = torch.all(sparse_mask.view(*sparse_mask.shape[:-2], -1), dim=-1)
    c = torch.count_nonzero(sparse_mask.view(*sparse_mask.shape[:-2], -1), dim=-1) >= count
    return torch.logical_or(a, c)

def make_block_sparse_if_count(count):
    # VIA CLOSURE
    def block_sparse_if_count_impl(sparse_mask : torch.Tensor, block_size : int):
        assert block_size == sparse_mask.shape[-1]
        assert block_size == sparse_mask.shape[-2]
        a = torch.all(sparse_mask.view(*sparse_mask.shape[:-2], -1), dim=-1)
        c = torch.count_nonzero(sparse_mask.view(*sparse_mask.shape[:-2], -1), dim=-1) >= count
        return torch.logical_or(a, c)
    
    return block_sparse_if_count_impl

class SparseBlockBatch():
    def __init__(self, block_size : int, dense : torch.Tensor, sparse_mask : torch.Tensor, sparse_criteria=block_sparse_if_all) -> None:
        '''
        Assumes dense = B x C x H x W
        Assumes mask  = B x H x W
        '''
        super().__init__()
        if dense.dim() == 3: dense = dense.unsqueeze(0)
        if sparse_mask.dim() == 2: sparse_mask = sparse_mask.unsqueeze(0)

        self.block_size = block_size
        self.dense_shape = dense.shape
        assert dense.shape[2] % block_size == 0 and dense.shape[3] % block_size == 0

        self.block_shape = dense.shape[0], dense.shape[1], dense.shape[2] // block_size, dense.shape[3] // block_size
        
        sparse_mask = sparse_mask.view(sparse_mask.shape[0],
                                      sparse_mask.shape[1]//block_size, block_size,
                                      sparse_mask.shape[2]//block_size, block_size).permute(0, 1, 3, 2, 4).reshape(-1, block_size, block_size)

        sparse_block_decision = sparse_criteria(sparse_mask, block_size)

        old_flat_len = dense.shape[0] * (dense.shape[2]//block_size) * (dense.shape[3]//block_size)

        dense = dense.view(dense.shape[0], dense.shape[1],
                                      dense.shape[2]//block_size, block_size,
                                      dense.shape[3]//block_size, block_size).permute(0, 2, 4, 1, 3, 5).reshape(-1, dense.shape[1], block_size, block_size)

        self.dense_flat_len = dense.shape[0]
        assert self.dense_flat_len == old_flat_len

        self.block_inds = torch.nonzero(sparse_block_decision).squeeze()
        self.blocks = torch.index_select(dense, 0, self.block_inds)

    def get_blocks(self):
        return self.blocks

    def to_dense(self, x):
        assert x.shape[-1] == x.shape[-2]
        assert self.block_size % x.shape[-1] == 0

        bs = self.block_shape # B x C x H* x W*
        dense = torch.zeros(self.dense_flat_len, x.shape[-3], x.shape[-2], x.shape[-1], device=x.device)
        # populate the dense matrix
        dense = dense.index_add(0, self.block_inds, x)
        # return it in contiguous form [B*(H/H*)*(W/W*), C, H*, W*] -> [B, C, H, W]
        dense = dense.reshape(self.block_shape[0], self.block_shape[2], self.block_shape[3], x.shape[-3], x.shape[-2], x.shape[-1])
        # [B, C, BH, H*, BW, W*] -> [B, C, H, W]
        dense = dense.permute(0, 3, 1, 4, 2, 5).reshape(-1, x.shape[-3], self.block_shape[2]*x.shape[-2], self.block_shape[3]*x.shape[-1])
        dense = dense.contiguous()

        return dense


class SparseMiniFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
        sparse_type="count",
        sparse_block_type="2_up",
        sparse_block_size=64,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SparseMiniFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        assert strides[-1] <= sparse_block_size
        self._size_divisibility = max(strides[-1], sparse_block_size)
        print("SIZE DIV:", self._size_divisibility)
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        self._sparse_block_size = sparse_block_size
        self._sparse_block_type = sparse_block_type
        self._sparse_type = sparse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x, sp):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        if sp.dim() == 4: sp = sp.squeeze()
        if x.dim() == 3: x = x.unsqueeze(0)

        criteria = {
            "count_2": make_block_sparse_if_count(2)
        }

        block_sparse_batch = SparseBlockBatch(self._sparse_block_size, x, sp, criteria[self._sparse_block_type])
        x_blocks = block_sparse_batch.get_blocks()

        bottom_up_features = self.bottom_up(x_blocks)

        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        
        results = [block_sparse_batch.to_dense(r) for r in results]

        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }




@BACKBONE_REGISTRY.register()
def build_resnet_sparse_mini_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = SparseMiniFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        sparse_type=cfg.INPUT.SPARSE.TYPE,
        sparse_block_type=cfg.INPUT.SPARSE.BLOCK_TYPE,
        sparse_block_size=cfg.INPUT.SPARSE.BLOCK_SIZE,
    )
    return backbone



@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone