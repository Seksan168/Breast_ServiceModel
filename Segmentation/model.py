import glob
import os
import re

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder


# Architecture registry — maps a short CLI key to the smp architecture name and a
# human-readable label. The key is also used for the output folder names
# (checkpoints/<key>, logs/<key>) so each model saves into its own folder.
# All architectures share the same encoder/encoder_weights from config.yaml.
ARCH_REGISTRY: dict[str, dict[str, str]] = {
    "unet":          {"architecture": "Unet",          "name": "U-Net"},
    "unetpp":        {"architecture": "UnetPlusPlus",  "name": "U-Net++"},
    "manet":         {"architecture": "MAnet",         "name": "MAnet"},
    "deeplabv3plus": {"architecture": "DeepLabV3Plus", "name": "DeepLabV3+"},
    # Decoders below are paired with a transformer encoder (e.g. mit_b2) in
    # config_transformer.yaml. SegFormer/UPerNet are the canonical transformer
    # segmentation heads; FPN is a lightweight extra. Unet++/Linknet are NOT
    # listed because smp does not support them with mit_* encoders.
    "segformer":     {"architecture": "Segformer",     "name": "SegFormer"},
    "upernet":       {"architecture": "UPerNet",       "name": "UPerNet"},
    "fpn":           {"architecture": "FPN",           "name": "FPN"},
    # Extra smp decoders. PSPNet/PAN/FPN are multi-scale context heads; Linknet
    # is a lightweight residual decoder. Like Unet++, Linknet and PAN reject some
    # encoders (Linknet ✗ mit_*, PAN ✗ pvt_v2) — main.py's guard skips those.
    "pspnet":        {"architecture": "PSPNet",        "name": "PSPNet"},
    "pan":           {"architecture": "PAN",           "name": "PAN"},
    "linknet":       {"architecture": "Linknet",       "name": "LinkNet"},
    # Swin-Unet is NOT an smp model: transformer encoder AND transformer decoder.
    # build_model() dispatches it to swin_unet.py. encoder_name in the config is
    # ignored for the model itself (the Swin-Tiny backbone is fixed) and only
    # labels the output folder, so use encoder_name: "swin_tiny" there.
    "swinunet":      {"architecture": "SwinUnet",      "name": "Swin-Unet"},
}

# Named groups for the --arch flag. "transformers" runs a curated set of
# decoders that are all verified to work with a transformer encoder (set via
# config_transformer.yaml). Every model in the group then shares the same MiT
# backbone, so the comparison isolates the decoder head.
ARCH_GROUPS: dict[str, list[str]] = {
    "transformers": ["segformer", "upernet", "deeplabv3plus", "manet", "fpn"],
    # The decoders added on top of the original four — train just these on an
    # existing encoder (e.g. --arch new_decoders --config config_aug.yaml) to
    # extend the comparison without re-running the models you already have.
    "new_decoders": ["pspnet", "pan", "linknet", "fpn"],
}


def run_slug(arch_key: str, encoder_name: str, aug: bool = False) -> str:
    """Build the folder slug for a run so checkpoints/logs of different encoders
    never collide (e.g. ``unet_resnet34`` vs ``unet_efficientnet-b3``).

    The encoder name is included because the saved weights are encoder-specific:
    loading a resnet34 checkpoint into an efficientnet model fails. ``_aug`` is
    appended when augmentation is on so aug / no-aug runs stay separate too.
    """
    enc = re.sub(r"[^0-9A-Za-z._-]+", "-", encoder_name)
    slug = f"{arch_key}_{enc}"
    return slug + ("_aug" if aug else "")


# Short folder tags per non-default loss, so runs with a different loss don't
# overwrite the baseline (dice_bce) checkpoints/logs of the same arch+encoder.
_LOSS_TAGS = {"focal_tversky": "_ft"}


def loss_suffix(cfg: dict) -> str:
    """Folder suffix that keeps runs with a different loss from colliding.

    An explicit ``loss.tag`` wins (use it to separate variants of the SAME loss
    type, e.g. two Focal-Tversky settings); otherwise fall back to a per-type
    tag, and '' for the default dice_bce."""
    lc = cfg.get("loss", {})
    if lc.get("tag"):
        return "_" + lc["tag"]
    return _LOSS_TAGS.get(lc.get("type", "dice_bce"), "")


def find_best_checkpoint(ckpt_dir: str) -> str | None:
    """Return the best checkpoint in ``ckpt_dir``.

    Filenames now carry the dice score (e.g. ``best_dice0.8421_bs8.pth``), so we
    pick the ``best_*.pth`` with the highest dice. Falls back to a legacy
    ``best.pth`` if present.
    """
    matches = glob.glob(os.path.join(ckpt_dir, "best_dice*.pth"))
    if matches:
        def _dice(p: str) -> float:
            try:
                return float(os.path.basename(p).split("dice")[1].split("_")[0])
            except (IndexError, ValueError):
                return -1.0
        return max(matches, key=_dice)
    legacy = os.path.join(ckpt_dir, "best.pth")
    return legacy if os.path.exists(legacy) else None


class UnetPlusPlusReduced(nn.Module):
    """U-Net++ for encoders that emit a 0-channel placeholder stage.

    The 4-stage transformer / ConvNeXt backbones (``tu-pvt_v2_*``,
    ``tu-convnext_*``) start at stride 4, so smp pads their feature list with a
    0-channel stride-2 placeholder to reach the usual 5 stages. ``Unet`` tolerates
    that (its decoder never uses skip channels as conv *out*-channels), but
    ``UnetPlusPlusDecoder`` sets each dense node's out-channels to the skip
    channels — a 0 there builds a ``[0, C, 3, 3]`` conv and crashes at forward
    (``weight to be at least 1 at dimension 0``).

    Fix: drop the empty stage, build a 4-block dense decoder over the real
    features (output ends at stride 2), and bilinear-upsample the head output
    back to the input resolution.
    """

    def __init__(self, encoder, decoder, head, drop_indices):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = head
        self.drop_indices = set(drop_indices)

    def forward(self, x):
        feats = self.encoder(x)
        feats = [f for i, f in enumerate(feats) if i not in self.drop_indices]
        out = self.segmentation_head(self.decoder(feats))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        return out


# U-Net++ decoder channels, one per (reduced) stage. The standard smp default is
# (256, 128, 64, 32, 16) for a 5-stage encoder; we drop the last when the empty
# stride-2 stage is removed, leaving a 4-block decoder.
_UNETPP_DECODER_CHANNELS = (256, 128, 64, 32)


def _build_unetpp_reduced(cfg: dict, encoder) -> nn.Module:
    """Assemble a :class:`UnetPlusPlusReduced` from a prebuilt encoder."""
    m = cfg["model"]
    ch = list(encoder.out_channels)
    drop = [i for i, c in enumerate(ch) if c == 0]
    real = [c for i, c in enumerate(ch) if i not in drop]
    dec_ch = _UNETPP_DECODER_CHANNELS[: len(real) - 1]
    decoder = UnetPlusPlusDecoder(
        encoder_channels=real, decoder_channels=dec_ch, n_blocks=len(dec_ch),
    )
    head = SegmentationHead(dec_ch[-1], m["num_classes"], activation=m["activation"],
                            kernel_size=3)
    return UnetPlusPlusReduced(encoder, decoder, head, drop)


def build_model(cfg: dict) -> nn.Module:
    """Build a segmentation model (arch from cfg) with pre-trained encoder."""
    m = cfg["model"]
    # Swin-Unet lives outside smp (transformer encoder + transformer decoder).
    if m["architecture"] == "SwinUnet":
        from swin_unet import build_swin_unet
        return build_swin_unet(cfg)
    # U-Net++ crashes on timm backbones that pad their features with a 0-channel
    # stride-2 stage (pvt_v2, convnext). Probe only "tu-" encoders (the only ones
    # that can emit an empty stage); fall through to the standard build otherwise.
    if m["architecture"] == "UnetPlusPlus" and m["encoder_name"].startswith("tu-"):
        encoder = get_encoder(m["encoder_name"], in_channels=m["in_channels"],
                              depth=5, weights=m["encoder_weights"])
        if any(c == 0 for c in encoder.out_channels):
            return _build_unetpp_reduced(cfg, encoder)
    model = smp.create_model(
        arch=m["architecture"],
        encoder_name=m["encoder_name"],
        encoder_weights=m["encoder_weights"],
        in_channels=m["in_channels"],    # fixed: was missing from original config
        classes=m["num_classes"],
        activation=m["activation"],      # None → raw logits
    )
    return model


def encoder_supported(architecture: str, encoder_name: str) -> tuple[bool, str | None]:
    """Cheaply check whether smp supports the (architecture, encoder) pair.

    Some decoders cannot be paired with every encoder — e.g. ``UnetPlusPlus`` and
    ``Linknet`` reject the transformer ``mit_*`` backbones — and smp only raises
    at ``create_model`` time. This builds the model with ``encoder_weights=None``
    (so no pretrained download happens) purely to trigger that validation, then
    throws the model away. Returns ``(ok, error_message)``.

    Swin-Unet is not an smp model: its Swin-Tiny backbone is fixed and ignores
    ``encoder_name``. To stop ``--arch all`` from retraining the same Swin-Tiny
    under every encoder's folder, it is only "supported" when ``encoder_name``
    actually names a Swin backbone (use config_swinunet.yaml, encoder swin_tiny).
    """
    if architecture == "SwinUnet":
        if "swin" in encoder_name.lower():
            return True, None
        return False, (f"SwinUnet has a fixed Swin-Tiny backbone; run it via "
                       f"config_swinunet.yaml (encoder_name=swin_tiny), not '{encoder_name}'")
    try:
        smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def count_parameters(model: nn.Module) -> float:
    """Return total trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint '{path}' (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metric_value: float,
    path: str,
    scaler=None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metric_value": metric_value,
        },
        path,
    )
