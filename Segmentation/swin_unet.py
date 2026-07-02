"""Swin-Unet — a transformer encoder + transformer decoder segmentation model.

This is the one architecture in this project whose *decoder* is a real
transformer (stacked Swin self-attention blocks), unlike the smp models whose
decoders are CNN/MLP. Design follows Cao et al., "Swin-Unet" (2021):

    encoder : ImageNet-pretrained Swin-Tiny (via timm, features_only)
    decoder : symmetric Swin stages with PatchExpand up-sampling + skip
              connections, then a x4 PatchExpand back to full resolution.

It is kept self-contained (own Swin blocks, no smp) and is wired into the rest
of the pipeline through build_model() in model.py, so train.py / dataset.py /
main.py use it unchanged via:

    python main.py --arch swinunet --config config_swinunet.yaml

Notes
-----
* The encoder is pretrained on ImageNet; the transformer decoder is trained
  from scratch (decoders always are).
* window_size is 8 so it evenly divides the decoder feature-map sizes for a
  512 input (128/64/32 are all divisible by 8). Change img_size and it must
  stay divisible by 32 and by (4 * window_size).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window attention building blocks (standard Swin Transformer)
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window, with relative position bias."""

    def __init__(self, dim: int, ws: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing="ij"))   # 2, ws, ws
        coords_flat = torch.flatten(coords, 1)                    # 2, ws*ws
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]   # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """One Swin Transformer block (W-MSA or SW-MSA + MLP), token input (B, L, C)."""

    def __init__(self, dim, input_resolution, num_heads, ws=8, shift=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.ws = ws
        self.shift = shift
        H, W = input_resolution
        if min(H, W) <= ws:        # window covers whole map -> no shift, full window
            self.shift = 0
            self.ws = min(H, W)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

        if self.shift > 0:
            self.register_buffer("attn_mask", self._build_mask(H, W))
        else:
            self.attn_mask = None

    def _build_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        slices = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.ws).view(-1, self.ws * self.ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        x_windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        attn_windows = self.attn(x_windows, self.attn_mask)
        attn_windows = attn_windows.view(-1, self.ws, self.ws, C)
        x = window_reverse(attn_windows, self.ws, H, W)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Up-sampling (PatchExpand) and decoder stage
# ---------------------------------------------------------------------------
class PatchExpand(nn.Module):
    """Double the spatial resolution, halve the channels (token in/out)."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape                       # C = 2*dim
        x = x.view(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C // 4)
        x = x.view(B, H * 2 * W * 2, C // 4)
        return self.norm(x)


class FinalPatchExpand_X4(nn.Module):
    """x4 up-sample back to the input resolution (token in, token out, same dim)."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape                       # C = 16*dim
        x = x.view(B, H, W, 4, 4, C // 16)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 4, W * 4, C // 16)
        x = x.view(B, H * 4 * W * 4, self.dim)
        return self.norm(x)


class DecoderStage(nn.Module):
    """PatchExpand up-sample, concat the encoder skip, fuse, then Swin blocks."""

    def __init__(self, dim, skip_dim, out_res, num_heads, depth=2, ws=8):
        super().__init__()
        # up_res is half the output resolution; expand brings it to out_res
        up_res = (out_res[0] // 2, out_res[1] // 2)
        self.up = PatchExpand(up_res, dim)             # dim -> dim//2, res x2
        self.concat_dim = dim // 2 + skip_dim
        self.reduce = nn.Linear(self.concat_dim, dim // 2)
        self.blocks = nn.ModuleList([
            SwinBlock(dim // 2, out_res, num_heads, ws=ws,
                      shift=0 if (i % 2 == 0) else ws // 2)
            for i in range(depth)
        ])

    def forward(self, x, skip):
        x = self.up(x)                                 # B, L, dim//2
        x = torch.cat([x, skip], dim=-1)               # B, L, concat_dim
        x = self.reduce(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class SwinUnet(nn.Module):
    def __init__(self, img_size=512, in_channels=3, num_classes=1,
                 pretrained=True, embed_dim=96, ws=8):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained, features_only=True,
            img_size=img_size, in_chans=in_channels,
        )
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]  # 96,192,384,768
        heads = [3, 6, 12, 24]
        p = img_size // 4                                # finest feature side (e.g. 128)
        res = [(p, p), (p // 2, p // 2), (p // 4, p // 4), (p // 8, p // 8)]

        # Decoder: 768@(16) -> 384@(32) -> 192@(64) -> 96@(128)
        self.dec3 = DecoderStage(dims[3], dims[2], res[2], heads[2], ws=ws)  # ->384 @32
        self.dec2 = DecoderStage(dims[2], dims[1], res[1], heads[1], ws=ws)  # ->192 @64
        self.dec1 = DecoderStage(dims[1], dims[0], res[0], heads[0], ws=ws)  # ->96  @128

        self.norm = nn.LayerNorm(dims[0])
        self.final_expand = FinalPatchExpand_X4(res[0], dims[0])             # ->512
        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        self.img_size = img_size

    @staticmethod
    def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
        """timm swin features are NHWC -> (B, H*W, C)."""
        B, H, W, C = feat.shape
        return feat.reshape(B, H * W, C)

    def forward(self, x):
        feats = self.encoder(x)                         # list of 4, NHWC
        e0, e1, e2, e3 = [self._to_tokens(f) for f in feats]

        d = self.dec3(e3, e2)                            # 384 @ 32x32
        d = self.dec2(d, e1)                             # 192 @ 64x64
        d = self.dec1(d, e0)                             # 96  @ 128x128

        d = self.norm(d)
        d = self.final_expand(d)                         # 96  @ 512x512 (tokens)
        H = W = self.img_size
        d = d.view(d.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.head(d)                              # B, num_classes, H, W


def build_swin_unet(cfg: dict) -> nn.Module:
    m = cfg["model"]
    img = cfg["data"].get("patch_size") or cfg["data"]["image_size"]
    return SwinUnet(
        img_size=img,
        in_channels=m["in_channels"],
        num_classes=m["num_classes"],
        pretrained=(m.get("encoder_weights") == "imagenet"),
    )
