"""
RetriVAD Qualitative Figure Generator — v5 (ACM MM 2026)
=========================================================
Matches UniVAD Figure 5 style:
  - Pure heatmap in column 4 (NOT blended with test image)
  - Clean white background, no title baked in
  - Tight spacing, professional grid
  - 300 DPI publication quality
  - ChestXray: multi-crop bank for stronger signal

Command (single line for Anaconda):
  python generate_heatmaps.py --mvtec_normal "..." --mvtec_test "..." --mvtec_gt "..." --visa_normal "..." --visa_test "..." --visa_gt "..." --brain_normal "..." --brain_test "..." --chest_normal "..." --chest_test "..." --output "figures/fig_qualitative.png"
"""

import argparse, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_dinov2():
    import torch
    print("Loading DINOv2 ViT-B/14 ...")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                       pretrained=True, verbose=False)
    m.eval()
    return m, torch


def encode(model, torch, path):
    import torch.nn.functional as F
    img = Image.open(path).convert("RGB")
    t   = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out   = model.forward_features(t)
        patch = out["x_norm_patchtokens"].squeeze(0)
        patch = F.normalize(patch, dim=1)
    return patch   # (256, 768)


def heatmap_standard(model, torch, norm_path, test_path,
                     plo=5, phi=95):
    import torch.nn.functional as F
    np_ = encode(model, torch, norm_path)
    tp  = encode(model, torch, test_path)
    dists    = torch.cdist(tp, np_)
    min_dist = dists.min(dim=1).values.reshape(16, 16).cpu().numpy()
    lo, hi   = np.percentile(min_dist, plo), np.percentile(min_dist, phi)
    if hi - lo < 1e-8:
        return np.zeros((224, 224), np.float32)
    clipped = np.clip(min_dist, lo, hi)
    normed  = (clipped - lo) / (hi - lo)
    t = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, (224, 224), mode="bilinear",
                       align_corners=False).squeeze().numpy()
    return up.astype(np.float32)


def heatmap_chest(model, torch, norm_path, test_path):
    """Multi-crop normal bank for stronger pneumonia signal."""
    import torch.nn.functional as F
    norm_pil = Image.open(norm_path).convert("RGB")
    W, H = norm_pil.size
    crops = [
        norm_pil,
        norm_pil.crop((0,      0,      W//2,   H)),
        norm_pil.crop((W//2,   0,      W,      H)),
        norm_pil.crop((W//4,   H//4,   3*W//4, 3*H//4)),
        norm_pil.crop((0,      H//4,   W//2,   3*H//4)),
        norm_pil.crop((W//2,   H//4,   W,      3*H//4)),
    ]
    bank_parts = []
    for c in crops:
        t = preprocess(c).unsqueeze(0)
        with torch.no_grad():
            out = model.forward_features(t)
            p   = out["x_norm_patchtokens"].squeeze(0)
            p   = F.normalize(p, dim=1)
        bank_parts.append(p)
    bank = torch.cat(bank_parts, dim=0)   # (6*256, 768)

    tp = encode(model, torch, test_path)
    dists    = torch.cdist(tp, bank)
    min_dist = dists.min(dim=1).values.reshape(16, 16).cpu().numpy()

    lo = np.percentile(min_dist, 2)
    hi = np.percentile(min_dist, 85)
    if hi - lo < 1e-8:
        return np.zeros((224, 224), np.float32)
    clipped = np.clip(min_dist, lo, hi)
    normed  = np.power((clipped - lo) / (hi - lo), 0.5)  # gamma boost

    t  = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, (224, 224), mode="bilinear",
                       align_corners=False).squeeze().numpy()
    return up.astype(np.float32)


def load_img(path, gray=False, size=224):
    mode = "L" if gray else "RGB"
    img  = Image.open(path).convert(mode).resize((size, size), Image.LANCZOS)
    arr  = np.array(img)
    if gray:
        arr = np.stack([arr]*3, axis=-1)
    return arr


def load_mask(path, size=224):
    if not path or not os.path.exists(path):
        return None
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    a = (np.array(m) > 10).astype(np.uint8)
    return a if a.sum() > 0 else None


def make_gt(test_img, mask):
    """Red overlay where mask exists; faded image otherwise."""
    if mask is not None:
        r = test_img.copy().astype(float)
        m = mask.astype(bool)
        r[m, 0] = np.clip(r[m, 0]*0.15 + 235, 0, 255)
        r[m, 1] = np.clip(r[m, 1]*0.15 + 10,  0, 255)
        r[m, 2] = np.clip(r[m, 2]*0.15 + 10,  0, 255)
        return r.astype(np.uint8), True
    faded = (test_img.astype(float)*0.35 + 155).clip(0,255).astype(np.uint8)
    return faded, False


def render(rows, out_path, dpi=300):
    """
    Matches UniVAD Fig 5 layout:
    4 cols: Normal | Test | Ground Truth | Heatmap (pure, no blend)
    Row labels on left, column labels on top.
    """
    n   = len(rows)
    COL_TITLES = ["Normal reference", "Test image",
                  "Ground truth", "Anomaly heatmap"]
    cmap = matplotlib.colormaps.get_cmap("jet")

    fig = plt.figure(figsize=(4*2.6 + 0.9, n*2.6 + 0.4),
                     dpi=dpi, facecolor="white")

    # Reserve right strip for colorbar
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[4*2.6, 0.55],
                              wspace=0.02)
    gs = gridspec.GridSpecFromSubplotSpec(n, 4, subplot_spec=outer[0],
                                          hspace=0.04, wspace=0.04)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)]

    for c, title in enumerate(COL_TITLES):
        axes[0][c].set_title(title, fontsize=9, fontweight="bold",
                             pad=5, fontfamily="sans-serif")

    for r, row in enumerate(rows):
        # Col 0: normal reference
        axes[r][0].imshow(row["normal"])
        # Col 1: test image
        axes[r][1].imshow(row["test"])
        # Col 2: ground truth (mask overlay or faded)
        axes[r][2].imshow(row["gt"])
        if not row["has_mask"]:
            axes[r][2].text(
                112, 185, "Image-level\nlabel only",
                ha="center", va="center", fontsize=7.5,
                color="#333", style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          alpha=0.80, ec="#bbb", lw=0.5))
        # Col 3: PURE heatmap (jet, no image blend — UniVAD style)
        axes[r][3].imshow(row["heatmap"], cmap="jet",
                          vmin=0, vmax=1, interpolation="bilinear")

        for c in range(4):
            axes[r][c].axis("off")

        # Row label
        axes[r][0].set_ylabel(row["label"], fontsize=9,
                               fontweight="bold", labelpad=8,
                               rotation=90, va="center",
                               fontfamily="sans-serif")
        axes[r][0].yaxis.set_label_position("left")
        axes[r][0].yaxis.label.set_visible(True)

    # Colorbar in right strip
    cbar_ax = fig.add_subplot(outer[1])
    sm = plt.cm.ScalarMappable(cmap="jet",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Anomaly score", fontsize=8, labelpad=6)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=7.5)

    plt.subplots_adjust(left=0.10, right=0.89, top=0.94,
                        bottom=0.01)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[✓] Saved: {out_path}  ({dpi} DPI)")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mvtec_normal",  required=True)
    p.add_argument("--mvtec_test",    required=True)
    p.add_argument("--mvtec_gt",      default=None)
    p.add_argument("--visa_normal",   required=True)
    p.add_argument("--visa_test",     required=True)
    p.add_argument("--visa_gt",       default=None)
    p.add_argument("--brain_normal",  required=True)
    p.add_argument("--brain_test",    required=True)
    p.add_argument("--chest_normal",  required=True)
    p.add_argument("--chest_test",    required=True)
    p.add_argument("--output", default="figures/fig_qualitative.png")
    p.add_argument("--dpi",    type=int, default=300)
    return p.parse_args()


def main():
    args         = parse()
    model, torch = load_dinov2()

    cfgs = [
        dict(label="MVTec-AD\n(Industrial)", norm=args.mvtec_normal,
             test=args.mvtec_test, gt=args.mvtec_gt, gray=False, mode="std"),
        dict(label="VisA\n(Industrial)",     norm=args.visa_normal,
             test=args.visa_test,  gt=args.visa_gt,   gray=False, mode="std"),
        dict(label="BrainMRI\n(Medical)",    norm=args.brain_normal,
             test=args.brain_test, gt=None,            gray=True,  mode="std"),
        dict(label="ChestXray\n(Medical)",   norm=args.chest_normal,
             test=args.chest_test, gt=None,            gray=True,  mode="chest"),
    ]

    rows = []
    for c in cfgs:
        tag = c["label"].split("\n")[0]
        print(f"Processing {tag} ...")
        normal  = load_img(c["norm"], c["gray"])
        test    = load_img(c["test"], c["gray"])
        mask    = load_mask(c.get("gt"))
        hm      = (heatmap_chest(model, torch, c["norm"], c["test"])
                   if c["mode"] == "chest"
                   else heatmap_standard(model, torch, c["norm"], c["test"]))
        gt, has = make_gt(test, mask)
        rows.append(dict(label=c["label"], normal=normal, test=test,
                         gt=gt, has_mask=has, heatmap=hm))

    print("Rendering ...")
    render(rows, args.output, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()