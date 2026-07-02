# Best Model for API

_Selected: 2026-07-02 — ranked by **test-set** Dice / pixel F1 across all trained runs (not the validation dice in the checkpoint filename)._

## ⭐ Recommended: UnetPlusPlus + se_resnext50_32x4d (augmented)

Best single deployable model — highest test Dice and a strong, well-balanced F1.

| Metric | Value |
|--------|-------|
| Architecture | UnetPlusPlus + `se_resnext50_32x4d` encoder |
| **Test Dice** | **0.5853** (0.5858 @ default 0.50) |
| Test IoU | 0.4556 |
| Pixel Precision / Recall / **F1** | 0.6654 / 0.6617 / **0.6636** |
| Tuned threshold | **0.45** |
| Params | 50.99 M |
| Inference speed | ~39.8 ms/image |
| Checkpoint | `checkpoints/unetpp_se_resnext50_32x4d_aug/best_dice0.5395_bs8.pth` |
| Config | `config_seresnext_aug.yaml` |
| Report | `logs/unetpp_se_resnext50_32x4d_aug_dice0.5395_bs8/report.md` |

### Inference command
```powershell
python inference.py `
  --arch unetpp `
  --image_dir path/to/images/ `
  --checkpoint checkpoints\unetpp_se_resnext50_32x4d_aug\best_dice0.5395_bs8.pth `
  --config config_seresnext_aug.yaml --threshold 0.45
```

> Note: the filename says `dice0.5395` — that is the **validation** dice used to pick the epoch.
> On the held-out **test** set this model scores Dice **0.5853**, the best of any single model.

## Top 6 single models ranked by test Dice

| # | Architecture | Test Dice | F1 | Thr | Checkpoint |
|---|--------------|:--:|:--:|:--:|------------|
| 1 | UnetPlusPlus + se_resnext50_32x4d (aug) **⭐** | **0.5853** | 0.6636 | 0.45 | `checkpoints/unetpp_se_resnext50_32x4d_aug/best_dice0.5395_bs8.pth` |
| 2 | Linknet + se_resnext50_32x4d (aug) | 0.5818 | 0.6600 | 0.50 | `checkpoints/linknet_se_resnext50_32x4d_aug/best_dice0.5454_bs8.pth` |
| 3 | MAnet + mit_b2 (aug) | 0.5748 | 0.4474 ⚠ | 0.30 | `checkpoints/manet_mit_b2_aug/best_dice0.5771_bs8.pth` |
| 4 | SwinUnet + swin_tiny (aug) | 0.5747 | 0.6282 | 0.05 | `checkpoints/swinunet_swin_tiny_aug/best_dice0.5743_bs4.pth` |
| 5 | UnetPlusPlus + resnet50 (aug) | 0.5718 | 0.6560 | 0.05 | `checkpoints/unetpp_resnet50_aug/best_dice0.5360_bs8.pth` |
| 6 | UnetPlusPlus + se_resnext50_32x4d (aug, `ftn`) | 0.5707 | **0.6765** | 0.25 | `checkpoints/unetpp_se_resnext50_32x4d_aug_ftn/best_dice0.5596_bs8.pth` |

> ⚠ **#3 MAnet + mit_b2** has a high Dice but a much lower pixel F1 (0.4474) — its
> precision/recall are imbalanced, so avoid it for the API despite the Dice ranking.
> Configs: #1/#2 `config_seresnext_aug.yaml`, #3 `config_transformer_aug.yaml`,
> #4 `config_swinunet_aug.yaml`, #5 `config_aug.yaml`, #6 `config_seresnext_aug_ftn.yaml`.

## Alternatives

| Goal | Model | Test Dice | F1 |
|------|-------|:--:|:--:|
| Best single Dice **(recommended)** | UnetPlusPlus + se_resnext50_32x4d (aug) | **0.5853** | 0.6636 |
| Best single F1 | UnetPlusPlus + se_resnext50_32x4d (aug, `ftn`) | 0.5707 | **0.6765** |
| Best overall (heavier) | 5-model ensemble + TTA | **0.6021** | 0.7010 |

- **Highest F1**, if precision/recall balance matters more than Dice: the fine-tuned
  `unetpp_se_resnext50_32x4d_aug_ftn` variant — F1 **0.6765**, Dice 0.5707.
- **Absolute best accuracy**, if latency/complexity is acceptable: the equal-weight
  **ensemble + TTA** of 5 models (Dice **0.6021**, F1 **0.7010**, threshold 0.35).
  See `logs/ensemble_tta/report.md`. Runs 5 networks × 4 TTA views per image, so it is
  much slower — not ideal for a low-latency API.

**Bottom line for the API:** ship **UnetPlusPlus + se_resnext50_32x4d (aug)** at threshold 0.45.
It gives the best single-model Dice with one fast forward pass. Switch to the `ftn`
variant if you want to maximize F1 instead, or the ensemble if accuracy outweighs speed.
