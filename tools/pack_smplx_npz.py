#!/usr/bin/env python3
"""Group transfer_model SMPL-X .pkl outputs into per-frame .npz files."""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _reshape_pose(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):  # rotation matrices
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 3 and arr.shape[-1] == 3:  # axis-angle
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def load_pkl(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    def get(key, alt=None):
        return data[key] if key in data else data.get(alt)

    out = {}
    out["betas"] = _to_numpy(get("betas"))
    out["global_orient"] = _reshape_pose(_to_numpy(get("global_orient")))
    out["body_pose"] = _reshape_pose(_to_numpy(get("body_pose")))
    out["transl"] = _to_numpy(get("transl", "translation"))
    out["left_hand_pose"] = _reshape_pose(_to_numpy(get("left_hand_pose")))
    out["right_hand_pose"] = _reshape_pose(_to_numpy(get("right_hand_pose")))
    out["jaw_pose"] = _reshape_pose(_to_numpy(get("jaw_pose")))
    out["leye_pose"] = _reshape_pose(_to_numpy(get("leye_pose")))
    out["reye_pose"] = _reshape_pose(_to_numpy(get("reye_pose")))
    out["expression"] = _to_numpy(get("expression"))
    out["joints_3d"] = _to_numpy(get("joints"))
    out["verts"] = _to_numpy(get("vertices"))
    return out


def group_by_frame(pkl_folder: Path) -> Dict[str, List[Path]]:
    frame_dict: Dict[str, List[Path]] = defaultdict(list)
    for pkl_path in pkl_folder.glob("*.pkl"):
        m = re.match(r"(.*)_p(\d+)$", pkl_path.stem)
        if not m:
            continue
        frame_id = m.group(1)
        frame_dict[frame_id].append(pkl_path)
    return frame_dict


def save_npz_per_frame(pkl_folder: Path, out_folder: Path) -> None:
    frames = group_by_frame(pkl_folder)
    if not frames:
        raise FileNotFoundError(f"No *_pXX.pkl files found under {pkl_folder}")

    out_folder.mkdir(parents=True, exist_ok=True)

    for frame_id, paths in tqdm(sorted(frames.items()), desc="Packing npz"):
        entries = []
        paths_sorted = sorted(paths, key=lambda p: int(re.match(r".*_p(\d+)$", p.stem).group(1)))
        for p in paths_sorted:
            entries.append(load_pkl(p))

        def stack(key):
            return np.stack([np.array(e[key]).squeeze() for e in entries], axis=0)

        verts = stack("verts")
        contact = np.zeros((verts.shape[0], verts.shape[1]), dtype=np.float32)

        np.savez(
            out_folder / f"{frame_id}.npz",
            betas=stack("betas"),
            body_pose=stack("body_pose"),
            global_orient=stack("global_orient"),
            transl=stack("transl"),
            left_hand_pose=stack("left_hand_pose"),
            right_hand_pose=stack("right_hand_pose"),
            jaw_pose=stack("jaw_pose"),
            leye_pose=stack("leye_pose"),
            reye_pose=stack("reye_pose"),
            expression=stack("expression"),
            joints_3d=stack("joints_3d"),
            verts=verts,
            contact=contact,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack SMPL-X pkl outputs to per-frame npz files."
    )
    parser.add_argument(
        "--pkl-folder",
        type=Path,
        required=True,
        help="Folder containing transfer_model output pkl files.",
    )
    parser.add_argument(
        "--out-folder",
        type=Path,
        required=True,
        help="Destination folder for npz files (e.g., the target smplx folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_npz_per_frame(args.pkl_folder, args.out_folder)


if __name__ == "__main__":
    main()
