# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL.Image import Image
from utils.transforms import SAM2Transforms, trunc_normal_
# import onnxruntime as ort
import axengine as ort
import cv2
import os

class ImagePredictor:
    def __init__(
        self,
        model_path,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        resolution=1024,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()

        print("Loading EdgeTAM Onnx models...")
        self.image_encoder = ort.InferenceSession(f"{model_path}/edgetam_image_encoder.axmodel")
        self.prompt_encoder = ort.InferenceSession(f"{model_path}/edgetam_prompt_encoder.axmodel")
        self.prompt_mask_encoder = ort.InferenceSession(f"{model_path}/edgetam_prompt_mask_encoder.axmodel")
        self.mask_decoder = ort.InferenceSession(f"{model_path}/edgetam_mask_decoder.axmodel")

        self.model_path = model_path

        self._transforms = SAM2Transforms(
            resolution=resolution,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold
        self.num_feature_levels = 3
        self.no_mem_embed = np.zeros((1, 1, 256))
        trunc_normal_(self.no_mem_embed, std=0.02)

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]

        input_image = self._transforms(image).astype(np.float32) # return 3xHxW np.ndarray
        input_image = input_image[None, ...]
        # np.save(f"{self.path}/input_image.npy", input_image)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        vision_feats  = self.image_encoder.run(None, {"input_image": input_image.astype(np.float32)})

        feats = [
                    np.transpose(feat[:, 0, :].reshape(H, W, feat.shape[-1]), (2, 0, 1))[np.newaxis, :]
                    for feat, (H, W) in zip(reversed(vision_feats), reversed(self._bb_feat_sizes))
                ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        #type check
        point_coords = point_coords.astype(np.float32) if point_coords is not None else None
        point_labels = point_labels.astype(np.float32) if point_labels is not None else None
        box = box.astype(np.float32) if box is not None else None
        mask_input = mask_input.astype(np.float32) if mask_input is not None else None

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks

        iou_predictions_np = iou_predictions[0]
        low_res_masks_np = low_res_masks[0]
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )

            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[np.newaxis, ...], point_labels[np.newaxis, ...]
        if box is not None:
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            if len(mask_logits.shape) == 3:
                mask_logits = mask_logits[np.newaxis, :, :, :]

        return mask_logits, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords,
        point_labels,
        boxes = None,
        mask_input = None,
        multimask_output = True,
        return_logits = False,
        img_idx = -1,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = np.array([[2, 3]], dtype=np.float32)
            box_labels = box_labels.repeat(boxes.shape[0], 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder 
            if concat_points is not None:
                concat_coords = np.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = np.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels)
            else:
                print("Only box input provided")
                concat_points = (box_coords, box_labels)

        # assert concat_points[0].shape[1] > 4, "only support points < 4"

        input_coords = np.tile(concat_points[0], (4, 1))[:, :4, :]
        input_labels = np.tile(concat_points[1], (4))[:, :4]

        # print("sparse_embeddings_tmp shape:", sparse_embeddings_tmp.shape)
        if mask_input.all() == 0:
            print("Get dense_embeddings_no_mask")
            sparse_embeddings = self.prompt_encoder.run(
            None,
            {
                "point_coords": input_coords if concat_points is not None else np.array([]),
                "point_labels": input_labels if concat_points is not None else np.array([])
                # "boxes": boxes if boxes is not None else np.zeros((1, 4), dtype=np.float32)
            },
        )[0]
            dense_embeddings = np.load(f"{self.model_path}/dense_embeddings_no_mask.npy")
        else:
            print("Get dense_embeddings_mask")
            sparse_embeddings = self.prompt_encoder.run(
            None,
            {
                "point_coords": input_coords if concat_points is not None else np.array([]),
                "point_labels": input_labels if concat_points is not None else np.array([])
                # "boxes": boxes if boxes is not None else np.zeros((1, 4), dtype=np.float32)
            },
        )[0]
            dense_embeddings = self.prompt_mask_encoder.run(
            None,
            {
                "input.1": mask_input
            },
        )[0]
        
        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction

        high_res_features = [
            feat_level[img_idx][np.newaxis, ...]
            for feat_level in self._features["high_res_feats"]
        ]

        low_res_masks, iou_predictions = self.mask_decoder.run(
            None,
            {
                "image_embeddings": self._features["image_embed"][img_idx][np.newaxis, ...],
                # "image_pe": image_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_feat_0": high_res_features[0],
                "high_res_feat_1": high_res_features[1],
                # "multimask_output": np.array([1 if multimask_output else 0], dtype=np.int32),
            },
        )

        # Upscale the masks to the original image resolution
        mask = low_res_masks[0].transpose(1, 2, 0)  # HxWxC
        resize_masks = cv2.resize(mask, (self._orig_hw[img_idx][1], self._orig_hw[img_idx][0]), interpolation=cv2.INTER_LINEAR)
        
        resize_masks = resize_masks[np.newaxis, ...]  # HxWx1xC
        resize_masks = np.clip(resize_masks, -32.0, 32.0) # 1xCxHxW
        
        if not return_logits:
            resize_masks = resize_masks > self.mask_threshold

        return resize_masks, iou_predictions, low_res_masks

    def get_image_embedding(self):
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1) for x in feature_maps]

        vision_pos_embeds = [x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
