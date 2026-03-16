"""Generate writeup-ready point-cloud figures from Waymo LiDAR data.

Uses sequence 3 (training_segment-10963653239323173269_1924_000_1944_000) as
specified in project-instructions/step-1/README.md for task ID_S1_EX2.
PCL is computed on-the-fly from the range image so no pre-cached .pkl files
are needed.

Usage:
    python misc/generate_pointcloud_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from tools.waymo_reader.simple_waymo_open_dataset_reader import (
    WaymoDataFileReader,
    dataset_pb2,
    label_pb2,
)
from misc.objdet_tools import pcl_from_range_image

# Sequence required by ID_S1_EX2
SEQ3_NAME = "training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels"
SEQ3_FRAME_RANGE = (0, 200)   # show_only_frames = [0, 200] per instructions
NUM_EXAMPLES = 10              # required by writeup instructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "img" / "writeup")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "img" / "writeup" / "vehicle_examples_summary.txt",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def _in_box_mask(pcl: np.ndarray, label) -> np.ndarray:
    """Boolean mask of points inside the 3-D axis-aligned rotated bounding box."""
    box = label.box
    xyz = pcl[:, :3]
    dx, dy, dz = xyz[:, 0] - box.center_x, xyz[:, 1] - box.center_y, xyz[:, 2] - box.center_z
    cos_h, sin_h = np.cos(box.heading), np.sin(box.heading)
    lon = dx * cos_h + dy * sin_h
    lat = -dx * sin_h + dy * cos_h
    return (
        (np.abs(lon) <= box.length / 2.0)
        & (np.abs(lat) <= box.width / 2.0)
        & (np.abs(dz) <= box.height / 2.0)
    )


def vehicle_local_cloud(
    pcl: np.ndarray, label
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (local_xyz, intensity, in_box_mask, dims) for the given vehicle label."""
    box = label.box
    xyz = pcl[:, :3]
    intensity = pcl[:, 3]
    dx, dy, dz = xyz[:, 0] - box.center_x, xyz[:, 1] - box.center_y, xyz[:, 2] - box.center_z
    cos_h, sin_h = np.cos(box.heading), np.sin(box.heading)
    lon = dx * cos_h + dy * sin_h
    lat = -dx * sin_h + dy * cos_h
    local_xyz = np.column_stack((lon, lat, dz))
    dims = np.array([box.length, box.width, box.height], dtype=float)
    in_box = (
        (np.abs(lon) <= box.length / 2.0)
        & (np.abs(lat) <= box.width / 2.0)
        & (np.abs(dz) <= box.height / 2.0)
    )
    return local_xyz, intensity, in_box, dims


def context_indices(local_xyz: np.ndarray, dims: np.ndarray, max_pts: int = 9000) -> np.ndarray:
    long_half = max(dims[0] * 1.8, 7.0)
    lat_half = max(dims[1] * 2.4, 4.5)
    mask = (
        (np.abs(local_xyz[:, 0]) <= long_half)
        & (np.abs(local_xyz[:, 1]) <= lat_half)
        & (local_xyz[:, 2] >= -1.6)
        & (local_xyz[:, 2] <= 2.8)
    )
    idx = np.where(mask)[0]
    if idx.size <= max_pts:
        return idx
    d = np.abs(local_xyz[idx, 0]) + 0.7 * np.abs(local_xyz[idx, 1])
    return idx[np.argsort(d)[:max_pts]]


def intensity_colors(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty((0, 4))
    upper = np.percentile(values, 99) if values.size > 5 else values.max()
    clipped = np.clip(values, 0.0, upper)
    denom = max(float(clipped.max() - clipped.min()), 1e-6)
    normalized = (clipped - clipped.min()) / denom
    return plt.cm.turbo(normalized)


def style_axis(
    ax,
    elev: float,
    azim: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0))


# ---------------------------------------------------------------------------
# Example selection from sequence 3
# ---------------------------------------------------------------------------

def _visibility_tag(pts: int) -> str:
    if pts == 0:
        return "occluded"
    if pts < 20:
        return "very sparse"
    if pts < 60:
        return "sparse"
    if pts < 150:
        return "moderate"
    if pts < 400:
        return "dense"
    return "very dense"


def select_diverse_examples(
    repo_root: Path,
    num_examples: int,
) -> list[dict[str, object]]:
    """Scan sequence 3 frames 0–200 and return num_examples diverse vehicle examples."""
    tfrecord = str(repo_root / "dataset" / f"{SEQ3_NAME}.tfrecord")
    reader = WaymoDataFileReader(tfrecord)

    candidates: list[tuple[int, int, int, float, object, np.ndarray]] = []
    for frame_idx, frame in enumerate(reader):
        if frame_idx < SEQ3_FRAME_RANGE[0]:
            continue
        if frame_idx > SEQ3_FRAME_RANGE[1]:
            break
        if not frame.laser_labels:
            continue

        pcl = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        for label_idx, label in enumerate(frame.laser_labels):
            if label.type != label_pb2.Label.Type.TYPE_VEHICLE:
                continue
            pts = int(np.sum(_in_box_mask(pcl, label)))
            dist = float(np.hypot(label.box.center_x, label.box.center_y))
            candidates.append((frame_idx, label_idx, pts, dist, frame, pcl))

    if not candidates:
        raise RuntimeError("No vehicle labels found in sequence 3 frames 0–200")

    # Sort by point count ascending; spread evenly to get diverse visibility
    candidates.sort(key=lambda c: c[2])
    step = max(1, len(candidates) // num_examples)
    chosen = list(range(0, len(candidates), step))[:num_examples]
    # Always include the densest vehicle
    if (len(candidates) - 1) not in chosen:
        chosen[-1] = len(candidates) - 1

    entries = []
    for ex_num, ci in enumerate(chosen, start=1):
        frame_idx, label_idx, pts, dist, frame, pcl = candidates[ci]
        entries.append(
            {
                "example": ex_num,
                "seq": SEQ3_NAME,
                "frame": frame_idx,
                "label": label_idx,
                "points": pts,
                "distance": dist,
                "visibility": _visibility_tag(pts),
                "_frame_obj": frame,
                "_pcl": pcl,
            }
        )
    return entries


def write_summary(entries: list[dict[str, object]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        for e in entries:
            fh.write(
                f"example={e['example']} seq={e['seq']} frame={e['frame']} "
                f"label={e['label']} points={e['points']} "
                f"distance_m={e['distance']:.2f} visibility={e['visibility']}\n"
            )
    print(f"Wrote summary to {summary_path}")


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_visibility_examples(
    output_path: Path,
    entries: list[dict[str, object]],
) -> None:
    n = len(entries)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 5.5, nrows * 4.5), dpi=180)

    for panel_idx, entry in enumerate(entries, start=1):
        label = entry["_frame_obj"].laser_labels[int(entry["label"])]
        pcl = entry["_pcl"]

        local_xyz, intensity, in_box, dims = vehicle_local_cloud(pcl, label)
        ctx_idx = context_indices(local_xyz, dims)
        in_box_ctx = np.where(in_box[ctx_idx])[0]
        sel_in_box = ctx_idx[in_box_ctx]

        ax = fig.add_subplot(nrows, ncols, panel_idx, projection="3d")
        ax.scatter(
            local_xyz[ctx_idx, 0], local_xyz[ctx_idx, 1], local_xyz[ctx_idx, 2],
            s=1.2, c="#d9d9d9", alpha=0.18, depthshade=False,
        )
        ax.scatter(
            local_xyz[sel_in_box, 0], local_xyz[sel_in_box, 1], local_xyz[sel_in_box, 2],
            s=3.0, c=intensity_colors(intensity[sel_in_box]), alpha=0.95, depthshade=False,
        )

        long_half = max(dims[0] * 1.3, 4.5)
        lat_half = max(dims[1] * 3.0, 5.0)
        style_axis(ax, 18, -62, (-long_half, long_half), (-lat_half, lat_half), (-1.2, 2.6))
        ax.set_box_aspect((1.8, 1.0, 0.75))
        ax.set_title(
            f"Ex {entry['example']}: {entry['visibility']}\n"
            f"pts={entry['points']}, dist={entry['distance']:.1f} m",
            fontsize=9, pad=4,
        )

    fig.suptitle(
        "Vehicle point-cloud examples — varying visibility\n"
        f"Sequence 3, frames {SEQ3_FRAME_RANGE[0]}–{SEQ3_FRAME_RANGE[1]}\n"
        "(gray = local context, colour = in-box LiDAR intensity)",
        fontsize=14, y=0.99,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.04, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved visibility examples to {output_path}")


def render_dense_vehicle_multiview(
    output_path: Path,
    entry: dict[str, object],
) -> None:
    label = entry["_frame_obj"].laser_labels[int(entry["label"])]
    pcl = entry["_pcl"]

    local_xyz, intensity, in_box, dims = vehicle_local_cloud(pcl, label)
    ctx_idx = context_indices(local_xyz, dims, max_pts=10000)
    sel_in_box = np.where(in_box & np.isin(np.arange(local_xyz.shape[0]), ctx_idx))[0]

    colors = intensity_colors(intensity[sel_in_box])
    views = [
        ("Oblique view", 18, -62),
        ("Front-quarter view", 14, 18),
        ("Side view", 8, -92),
        ("Top view", 88, -90),
    ]

    fig = plt.figure(figsize=(14, 8), dpi=180)
    for panel_idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, panel_idx, projection="3d")
        ax.scatter(
            local_xyz[ctx_idx, 0], local_xyz[ctx_idx, 1], local_xyz[ctx_idx, 2],
            s=1.2, c="#d9d9d9", alpha=0.18, depthshade=False,
        )
        ax.scatter(
            local_xyz[sel_in_box, 0], local_xyz[sel_in_box, 1], local_xyz[sel_in_box, 2],
            s=3.0, c=colors, alpha=0.95, depthshade=False,
        )
        style_axis(ax, elev, azim, (-4.2, 4.2), (-3.8, 3.8), (-1.2, 2.3))
        ax.set_box_aspect((1.5, 1.0, 0.8))
        ax.set_title(title, fontsize=12, pad=4)

    fig.suptitle(
        "Representative dense vehicle point cloud — Sequence 3\n"
        f"frame {entry['frame']}, {entry['points']} in-box pts, {entry['distance']:.1f} m",
        fontsize=14, y=0.97,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.05, hspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved multi-view dense cloud to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_range_image(output_path: Path, frame, lidar_name: int) -> None:
    """Export a stacked range+intensity image (replicating show_range_image logic)."""
    import cv2
    from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils

    lidar = waymo_utils.get(frame.lasers, lidar_name)
    range_image, _, _ = waymo_utils.parse_range_image_and_camera_projection(lidar)

    range_ch = np.clip(range_image[:, :, 0], 0.0, None)
    intens_ch = np.clip(range_image[:, :, 1], 0.0, None)

    rmax = range_ch.max()
    img_range = (range_ch / rmax * 255.0) if rmax > 0 else np.zeros_like(range_ch)

    iv = intens_ch[intens_ch > 0]
    if iv.size > 0:
        p1, p99 = np.percentile(iv, [1, 99])
        if p99 > p1:
            img_intens = np.clip(intens_ch, p1, p99)
            img_intens = (img_intens - p1) / (p99 - p1) * 255.0
        else:
            img_intens = np.zeros_like(intens_ch)
    else:
        img_intens = np.zeros_like(intens_ch)

    stacked = np.vstack((img_range, img_intens)).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), stacked)
    print(f"Saved range image to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Scanning sequence 3 frames {SEQ3_FRAME_RANGE[0]}–{SEQ3_FRAME_RANGE[1]} …")
    entries = select_diverse_examples(args.repo_root, NUM_EXAMPLES)
    write_summary(entries, args.summary_path)

    render_visibility_examples(
        args.output_dir / "vehicle_visibility_examples.png",
        entries,
    )

    # Use the densest vehicle for the multi-view detail figure
    representative = max(entries, key=lambda e: int(e["points"]))
    render_dense_vehicle_multiview(
        args.output_dir / "representative_vehicle_intensity_views.png",
        representative,
    )

    # Range image from the densest frame (sequence 3) — used to support
    # intensity findings per writeup requirement for ID_S1_EX2
    render_range_image(
        args.output_dir / "range_image_seq3_dense_frame.png",
        representative["_frame_obj"],
        dataset_pb2.LaserName.TOP,
    )


if __name__ == "__main__":
    main()
