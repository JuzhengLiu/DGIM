import argparse
from pathlib import Path

import torch
import sys
import cv2
import warnings
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

from src.config.paths import load_paths, resolve_path
from src.utils.checkpoint import load_matcher

warnings.filterwarnings("ignore")


def draw_matches(image0, image1, points0, points1):
    difference = image0.shape[0] - image1.shape[0]
    if difference < 0:
        top = abs(difference) // 2
        bottom = abs(difference) - top
        image0 = cv2.copyMakeBorder(image0, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points0) > 0:
            points0[:, 1] += top
    elif difference > 0:
        top = difference // 2
        bottom = difference - top
        image1 = cv2.copyMakeBorder(image1, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points1) > 0:
            points1[:, 1] += top
    points0 = [cv2.KeyPoint(point0[0], point0[1], 1) for point0 in points0]
    points1 = [cv2.KeyPoint(point1[0], point1[1], 1) for point1 in points1]
    matches = [cv2.DMatch(index, index, 1) for index in range(len(points0))]
    # noinspection PyTypeChecker
    matches_visual = cv2.drawMatches(image0, points0, image1, points1, matches, None, (0, 255, 0))
    return matches_visual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select two images and match them with DGIM. A CUDA GPU and desktop display are required. "
                    "Relative paths are resolved from the repository root.")
    parser.add_argument("--paths", help="Path configuration YML; defaults to configs/paths.yml with optional local overrides")
    parser.add_argument("--checkpoint", help="Override matcher_checkpoint in the path configuration")
    parser.add_argument("--output-dir", help="Override demo_output_dir for the saved match figure")
    args = parser.parse_args()

    try:
        paths = load_paths(args.paths)
        checkpoint_value = resolve_path(args.checkpoint) if args.checkpoint is not None else paths["matcher_checkpoint"]
        output_value = resolve_path(args.output_dir) if args.output_dir is not None else paths["demo_output_dir"]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(f"Cannot load paths: {exc}. Check the configuration selected by --paths.")

    checkpoint_path = Path(checkpoint_value) if checkpoint_value is not None else None
    sample_dir = Path(paths["sample_dir"]) if paths["sample_dir"] is not None else None
    output_dir = Path(output_value) if output_value is not None else None
    if checkpoint_path is None or not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}. Set matcher_checkpoint in the --paths "
                     "configuration or provide --checkpoint.")
    if sample_dir is not None and not sample_dir.is_dir():
        parser.error(f"Initial image directory not found: {sample_dir}. Set sample_dir in the --paths configuration.")
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"Cannot create output directory {output_dir}: {exc}. Check demo_output_dir or --output-dir.")

    half_precision = True  # default: False
    matcher = load_matcher(checkpoint_path, device='cuda', half=half_precision)

    root = tk.Tk()
    root.withdraw()

    image0_path = filedialog.askopenfilename(
        title="Select the Left Image (English-Only Path)",
        initialdir=str(sample_dir) if sample_dir is not None else None,
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")]
    )
    if not image0_path:
        root.destroy()
        sys.exit()

    image1_path = filedialog.askopenfilename(
        title="Select the Right Image (English-Only Path)",
        initialdir=str(sample_dir) if sample_dir is not None else None,
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")]
    )
    root.destroy()
    if not image1_path:
        sys.exit()

    # If matching is difficult, the input images can be resized proportionally,
    # and the output matches will be automatically resized back.
    scale_ratio0 = 1.0  # default: 1.0
    scale_ratio1 = 1.0  # default: 1.0

    image0_bgr = cv2.imread(image0_path)
    image1_bgr = cv2.imread(image1_path)
    if image0_bgr is None or image1_bgr is None:
        unreadable_path = image0_path if image0_bgr is None else image1_path
        parser.error(f"Cannot read selected image: {unreadable_path}. Select a readable image; "
                     "sample_dir in the --paths configuration controls the initial folder.")
    image0_rgb = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image0_gray = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
    image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

    height0, width0 = image0_gray.shape
    new_width0, new_height0 = map(lambda x: int(x * scale_ratio0 // 8 * 8), [width0, height0])
    image0_gray = cv2.resize(image0_gray, (new_width0, new_height0))
    scale0 = np.array([width0 / new_width0, height0 / new_height0], dtype=np.float16)

    height1, width1 = image1_gray.shape
    new_width1, new_height1 = map(lambda x: int(x * scale_ratio1 // 8 * 8), [width1, height1])
    image1_gray = cv2.resize(image1_gray, (new_width1, new_height1))
    scale1 = np.array([width1 / new_width1, height1 / new_height1], dtype=np.float16)

    image0 = torch.from_numpy(image0_gray)[None][None].cuda() / 255
    image1 = torch.from_numpy(image1_gray)[None][None].cuda() / 255
    if half_precision:
        image0 = image0.half()
        image1 = image1.half()
    batch = {"image0": image0, "image1": image1}

    with torch.no_grad():
        matcher(batch)
        matched_points0 = batch["mkpts0_f"].cpu().numpy() * scale0
        matched_points1 = batch["mkpts1_f"].cpu().numpy() * scale1
    torch.cuda.empty_cache()

    if len(matched_points0) >= 4:
        homography, mask = cv2.findHomography(matched_points1, matched_points0, cv2.USAC_MAGSAC, 5.0)
        if mask is not None:
            matched_points0 = matched_points0[mask.flatten() == 1]
            matched_points1 = matched_points1[mask.flatten() == 1]
        else:
            print("Homography estimation failed; displaying unfiltered matches.")

    matches_visual = draw_matches(image0_rgb, image1_rgb, matched_points0, matched_points1)

    figure = plt.figure()
    plt.imshow(matches_visual, "gray")
    plt.title(f"{len(matched_points0)} matches")
    plt.axis("off")
    if output_dir is not None:
        output_path = output_dir / f"{Path(image0_path).stem}__{Path(image1_path).stem}_matches.png"
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved matches: {output_path}")
    plt.show()
