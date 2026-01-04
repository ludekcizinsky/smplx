from __future__ import annotations

from dataclasses import dataclass
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
import tyro

# Local import with fallback for direct script execution
try:
    from transfer_model.utils import batch_rot2aa
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transfer_model.utils import batch_rot2aa


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _reshape_pose(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):  # rotation matrices
        flat = arr.reshape(-1, 3, 3)
        aa = batch_rot2aa(torch.from_numpy(flat)).cpu().numpy()
        return aa.reshape(arr.shape[0], arr.shape[1], 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:  # axis-angle
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 9:  # flattened rotmats
        flat = arr.reshape(-1, 3, 3)
        aa = batch_rot2aa(torch.from_numpy(flat)).cpu().numpy()
        return aa.reshape(arr.shape[0], 3)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def load_pkl(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    def get(key, alt=None):
        return data[key] if key in data else data.get(alt)

    out = {"_path": str(path)}
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

    # Squeeze leading singleton batch dims for per-sample storage
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.shape[0] == 1:
            out[k] = v[0]
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


def _infer_model_type(sample: Dict[str, np.ndarray]) -> str:
    if sample.get("left_hand_pose") is not None or sample.get("right_hand_pose") is not None:
        return "smplx"
    if sample.get("jaw_pose") is not None or sample.get("expression") is not None:
        return "smplx"
    return "smpl"


def save_npz_per_frame(pkl_folder: Path, out_folder: Path, model_type: str) -> None:
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
            values = []
            for entry in entries:
                val = entry.get(key)
                if val is None:
                    raise KeyError(f"Missing '{key}' in {entry.get('_path', 'entry')}")
                values.append(np.array(val))
            return np.stack(values, axis=0)

        verts = stack("verts")

        frame_type = model_type
        if frame_type == "auto":
            frame_type = _infer_model_type(entries[0])

        root_pose = stack("global_orient").reshape(len(entries), -1, 3)
        body_pose = stack("body_pose").reshape(len(entries), -1, 3)
        trans = stack("transl").reshape(len(entries), -1, 3)

        if frame_type == "smplx":
            jaw_pose = stack("jaw_pose").reshape(len(entries), -1, 3)
            leye_pose = stack("leye_pose").reshape(len(entries), -1, 3)
            reye_pose = stack("reye_pose").reshape(len(entries), -1, 3)
            lhand_pose = stack("left_hand_pose").reshape(len(entries), -1, 3)
            rhand_pose = stack("right_hand_pose").reshape(len(entries), -1, 3)

            np.savez(
                out_folder / f"{frame_id}.npz",
                betas=stack("betas")[..., :10],  # keep 10 betas
                root_pose=root_pose[:, 0],      # (P,3)
                body_pose=body_pose[:, :21],    # (P,21,3)
                jaw_pose=jaw_pose[:, 0],        # (P,3)
                leye_pose=leye_pose[:, 0],      # (P,3)
                reye_pose=reye_pose[:, 0],      # (P,3)
                lhand_pose=lhand_pose[:, :15],  # (P,15,3)
                rhand_pose=rhand_pose[:, :15],  # (P,15,3)
                trans=trans[:, 0],              # (P,3)
                expression=stack("expression"),
                joints_3d=stack("joints_3d"),
                verts=verts,
            )
        elif frame_type == "smpl":
            np.savez(
                out_folder / f"{frame_id}.npz",
                betas=stack("betas")[..., :10],  # keep 10 betas
                root_pose=root_pose[:, 0],      # (P,3)
                body_pose=body_pose[:, :23],    # (P,23,3)
                trans=trans[:, 0],              # (P,3)
                joints_3d=stack("joints_3d"),
                verts=verts,
            )
        else:
            raise ValueError(f"Unsupported model_type: {frame_type}. Expected 'smplx', 'smpl', or 'auto'.")


@dataclass
class Args:
    """CLI arguments for packing transfer_model outputs."""

    pkl_folder: Path
    out_folder: Path
    model_type: str = "auto"


def main() -> None:
    args = tyro.cli(Args, description="Pack transfer_model pkl outputs to per-frame npz files.")
    save_npz_per_frame(args.pkl_folder, args.out_folder, args.model_type)


if __name__ == "__main__":
    main()
