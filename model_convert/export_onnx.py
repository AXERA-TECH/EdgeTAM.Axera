#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Tuple
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Suppress warnings
warnings.filterwarnings("ignore", message="Torch version .* has not been tested with coremltools")
warnings.filterwarnings("ignore", message=".*resources.bin missing.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam2.build_sam import _load_checkpoint
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure you have installed EdgeTAM properly.")
    sys.exit(1)


class EdgeTAMImageEncoder(torch.nn.Module):
    """EdgeTAM Image Encoder wrapper for onnx export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.num_feature_levels = 3
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)

        backbone_out, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        return  vision_feats

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


class EdgeTAMPromptEncoder(torch.nn.Module):
    """EdgeTAM Prompt Encoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.image_embedding_size = sam_model.sam_prompt_encoder.image_embedding_size
        self.no_mask_embed = sam_model.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

    def forward(self,
                point_coords: torch.Tensor,
                point_labels: torch.Tensor):

        sparse_embeddings, _ = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        return sparse_embeddings

class EdgeTAMPromptWithMaskEncoder(torch.nn.Module):
    """EdgeTAM Prompt Encoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model

    def forward(self,
                point_coords: torch.Tensor,
                point_labels: torch.Tensor,
                mask_input: torch.Tensor):

        dense_embeddings = self.model.sam_prompt_encoder._embed_masks(mask_input)
        return dense_embeddings


class EdgeTAMMaskDecoder(torch.nn.Module):
    """EdgeTAM Mask Decoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        self.proper_image_pe = self.model.sam_prompt_encoder.get_dense_pe()

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                high_res_feat_0: torch.Tensor,
                high_res_feat_1: torch.Tensor,
                multimask_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        use_multimask = multimask_output[0].item() > 0.5
        high_res_features = [high_res_feat_0, high_res_feat_1]

        # Use proper position encoding from prompt encoder
        
        # print(proper_image_pe)

        sam_outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.proper_image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=use_multimask,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        masks = sam_outputs[0]
        iou_pred = sam_outputs[1]

        if not use_multimask:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        return masks, iou_pred


def export_image_encoder(model, output_path: str):
    """Export image encoder to CoreML."""
    print("Exporting Image Encoder...")
    encoder_wrapper = EdgeTAMImageEncoder(model)
    encoder_wrapper.eval()

    example_input = torch.randn(1, 3, 1024, 1024)

    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            example_input,
            output_path,
            opset_version=17,
            input_names=["input_image"],
            output_names=["vision_features", "high_res_feat_0", "high_res_feat_1"])

    print(f"  Saved to {output_path}")


def export_prompt_encoder(model, output_path: str):
    """Export prompt encoder to CoreML."""
    print("Exporting Prompt Encoder...")
    encoder_wrapper = EdgeTAMPromptEncoder(model)
    encoder_wrapper.eval()

    point_coords = torch.zeros(1, 4, 2, dtype=torch.float32)
    point_coords[0, 0] = torch.tensor([512.0, 512.0])
    point_coords[0, 1] = torch.tensor([256.0, 256.0])
    point_coords[0, 2] = torch.tensor([512.0, 512.0])
    point_coords[0, 3] = torch.tensor([256.0, 256.0])
    point_labels = torch.full((1, 4), -1, dtype=torch.float32)
    point_labels[0, 0] = 1.0
    point_labels[0, 1] = 0.0
    point_labels[0, 2] = 1.0
    point_labels[0, 3] = 0.0
    boxes = torch.zeros(1, 4, dtype=torch.float32)
    mask_input = torch.zeros(1, 1, 256, 256)

    np.save(f"{os.path.dirname(output_path)}/dense_embeddings_no_mask.npy", encoder_wrapper.no_mask_embed.detach().numpy())
    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (point_coords, point_labels),
            # (point_coords, point_labels, boxes),
            output_path,
            opset_version=17,
            # input_names=["point_coords", "point_labels", "boxes", "mask_input"],
            input_names=["point_coords", "point_labels"],
            output_names=["sparse_embeddings"])

    print(f"  Saved to {output_path}")

def export_prompt_mask_encoder(model, output_path: str):
    """Export prompt encoder to CoreML."""
    print("Exporting Prompt Encoder...")
    encoder_wrapper = EdgeTAMPromptWithMaskEncoder(model)
    encoder_wrapper.eval()

    point_coords = torch.zeros(1, 4, 2, dtype=torch.float32)
    point_coords[0, 0] = torch.tensor([512.0, 512.0])
    point_coords[0, 1] = torch.tensor([256.0, 256.0])
    point_coords[0, 2] = torch.tensor([512.0, 512.0])
    point_coords[0, 3] = torch.tensor([256.0, 256.0])
    point_labels = torch.full((1, 4), -1, dtype=torch.float32)
    point_labels[0, 0] = 1.0
    point_labels[0, 1] = 0.0
    point_labels[0, 2] = 1.0
    point_labels[0, 3] = 0.0
    boxes = torch.zeros(1, 4, dtype=torch.float32)
    mask_input = torch.zeros(1, 1, 256, 256)

    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (point_coords, point_labels, mask_input),
            # (point_coords, point_labels, boxes),
            output_path,
            opset_version=17,
            # input_names=["point_coords", "point_labels", "boxes", "mask_input"],
            input_names=["mask_input"],
            output_names=["dense_embeddings"])

    print(f"  Saved to {output_path}")


def export_mask_decoder(model, output_path: str):
    """Export mask decoder to CoreML."""
    print("Exporting Mask Decoder...")
    decoder_wrapper = EdgeTAMMaskDecoder(model)
    decoder_wrapper.eval()

    image_embeddings = torch.randn(1, 256, 64, 64)
    image_pe = model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings = torch.randn(1, 5, 256)
    dense_prompt_embeddings = torch.randn(1, 256, 64, 64)
    high_res_feat_0 = torch.randn(1, 32, 256, 256)
    high_res_feat_1 = torch.randn(1, 64, 128, 128)
    multimask_output = torch.tensor([False])

    print("Image PE shape:", image_pe[0].size())
    with torch.no_grad():
        torch.onnx.export(
            decoder_wrapper,
            (
                image_embeddings,
                image_pe[0],
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                high_res_feat_0,
                high_res_feat_1,
                multimask_output
            ),
            output_path,
            opset_version=17,
            input_names=[
                "image_embeddings",
                "image_pe",
                "sparse_prompt_embeddings",
                "dense_prompt_embeddings",
                "high_res_feat_0",
                "high_res_feat_1",
                "multimask_output"
            ],
            output_names=["masks", "iou_pred"])
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export EdgeTAM to onnx")
    parser.add_argument("--sam2_cfg", default='./configs/edgetam.yaml', help="Path to EdgeTAM config file")
    parser.add_argument("--sam2_checkpoint", default='./checkpoints/edgetam.pt', help="Path to EdgeTAM checkpoint")
    parser.add_argument("--output_dir", default="./onnx_models", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading EdgeTAM model...")
    try:
        device = torch.device("cpu")
        GlobalHydra.instance().clear()

        config_path = os.path.abspath(args.sam2_cfg)
        config_dir = os.path.dirname(config_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            OmegaConf.resolve(cfg)

            model = instantiate(cfg.model, _recursive_=True)

            if args.sam2_checkpoint:
                _load_checkpoint(model, args.sam2_checkpoint)

            model = model.to(device)
            model.eval()

    except Exception as e:
        print(f"Failed to load EdgeTAM model: {e}")
        sys.exit(1)

    print(f"\nExporting to {args.output_dir}...")

    # try:
    export_image_encoder(
        model,
        os.path.join(args.output_dir, "edgetam_image_encoder.onnx")
    )

    export_prompt_encoder(
        model,
        os.path.join(args.output_dir, "edgetam_prompt_encoder.onnx")
    )

    export_prompt_mask_encoder(
        model,
        os.path.join(args.output_dir, "edgetam_prompt_mask_encoder.onnx")
    )

    export_mask_decoder(
        model,
        os.path.join(args.output_dir, "edgetam_mask_decoder.onnx")
    )

    print(f"\nExport completed successfully!")
    print(f"Models saved to: {args.output_dir}")

    # except Exception as e:
    #     print(f"Export failed: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()