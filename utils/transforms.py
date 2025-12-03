# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import albumentations as A
import numpy as np
from scipy.stats import truncnorm
import cv2

class SAM2Transforms():
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0, onnx=False
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.transforms = A.Compose([
                                    A.Resize(height=resolution, width=resolution),  # 先 resize
                                    A.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet RGB mean
                                                std=[0.229, 0.224, 0.225],    # ImageNet RGB std
                                                max_pixel_value=255.0,        # 因为输入是 0-255 的 uint8
                                                p=1.0)
                                    ])
        self.onnx = onnx

    def __call__(self, x):
        #x: np.ndarray, HWC, uint8, RGB
        # x_normal = cv2.resize(x, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        if self.onnx:
            x_normal = self.transforms(image=x)['image']
            return x_normal.transpose(2, 0, 1)
        else:
            x_normal = cv2.resize(x, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            return x_normal.transpose(2, 0, 1)
        

    def forward_batch(self, img_list):
        #img_list: list of np.ndarray, HWC, uint8, RGB
        img_batch = [self.transforms(img) for img in img_list]
        img_batch = np.concatenate([img[np.newaxis, :].transpose(0, 3, 1, 2) for img in img_batch], axis=0)
        return img_batch

    def transform_coords(
        self, coords, normalize=False, orig_hw=None
    ):
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.copy()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        coords = coords * self.resolution 
        return coords

    def transform_boxes(
        self, boxes, normalize=False, orig_hw=None
    ):
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    """
    def postprocess_masks(self, masks, orig_hw):
        # Perform PostProcessing on output masks.
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
    """

def trunc_normal_(arr, std=0.02, mean=0.0):
    """
    用截断正态分布原地初始化 numpy array
    
    截断范围: [mean - 2*std, mean + 2*std]
    """
    # 计算截断边界（以标准差为单位）
    a = (mean - 2 * std - mean) / std  # = -2
    b = (mean + 2 * std - mean) / std  # = +2
    
    # 生成截断正态分布样本
    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=arr.shape)
    
    # 原地赋值
    arr[:] = samples