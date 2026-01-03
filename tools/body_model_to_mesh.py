#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import smplx
import trimesh
from tqdm import tqdm
import tyro


@dataclass
class Args:
    """CLI arguments for exporting SMPL/SMPL-X meshes."""

    dataset_root: Path
    obj_output: Path
    model_folder: Path
    model_type: str = "smpl"
    gender: str = "neutral"
    model_ext: str = "npz"


def _load_faces(model_folder: Path, model_type: str, gender: str, ext: str) -> np.ndarray:
    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        ext=ext,
    )
    return np.asarray(model.faces, dtype=np.int32)


def export_npz_to_obj(
    dataset_root: Path,
    obj_output: Path,
    model_folder: Path,
    model_type: str = "smpl",
    gender: str = "neutral",
    model_ext: str = "npz",
) -> Sequence[Path]:
    """Convert all SMPL/SMPL-X npz files in dataset_root to obj files in obj_output."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    if model_type not in {"smpl", "smplx"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'smpl' or 'smplx'.")

    obj_output.mkdir(parents=True, exist_ok=True)
    faces = _load_faces(model_folder, model_type, gender, model_ext)

    npz_files = sorted(dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {dataset_root}")

    written = []
    for npz_path in tqdm(npz_files, desc="Exporting SMPL meshes"):
        data = np.load(npz_path, allow_pickle=True)
        if "verts" not in data:
            raise KeyError(f"'verts' key missing in {npz_path}")
        verts = data["verts"]
        if verts.ndim != 3:
            raise ValueError(f"Unexpected verts shape {verts.shape} in {npz_path}")

        num_people = verts.shape[0]
        for person_idx in range(num_people):
            out_path = obj_output / f"{npz_path.stem}_p{person_idx:02d}.obj"
            mesh = trimesh.Trimesh(
                verts[person_idx], faces, process=False
            )
            mesh.export(out_path)
            written.append(out_path)

    return written


def main() -> None:
    args = tyro.cli(Args, description="Export SMPL/SMPL-X npz frames to obj meshes.")
    written = export_npz_to_obj(
        dataset_root=args.dataset_root,
        obj_output=args.obj_output,
        model_folder=args.model_folder,
        model_type=args.model_type,
        gender=args.gender,
        model_ext=args.model_ext,
    )
    print(f"Wrote {len(written)} meshes to {args.obj_output}")


if __name__ == "__main__":
    main()
