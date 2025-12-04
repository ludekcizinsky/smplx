#!/usr/bin/env python3
"""Export HI4D SMPL npz frames to SMPL meshes for transfer_model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import smplx
import trimesh
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path(
    "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/fight00/smpl"
)
DEFAULT_OBJ_OUTPUT = Path(
    "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/fight00/smpl_to_smplx_meshes"
)
DEFAULT_BODY_MODEL_FOLDER = Path("/home/cizinsky/body_models")


def _load_faces(model_folder: Path, gender: str, ext: str) -> np.ndarray:
    model = smplx.create(
        model_folder,
        model_type="smpl",
        gender=gender,
        ext=ext,
    )
    return np.asarray(model.faces, dtype=np.int32)


def export_npz_to_obj(
    dataset_root: Path,
    obj_output: Path,
    model_folder: Path,
    gender: str = "neutral",
    model_ext: str = "npz",
) -> Sequence[Path]:
    """Convert all SMPL npz files in dataset_root to obj files in obj_output."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    obj_output.mkdir(parents=True, exist_ok=True)
    faces = _load_faces(model_folder, gender, model_ext)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HI4D SMPL npz frames to SMPL obj meshes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Folder with SMPL npz files (HI4D per-frame files).",
    )
    parser.add_argument(
        "--obj-output",
        type=Path,
        default=DEFAULT_OBJ_OUTPUT,
        help="Destination folder for exported obj meshes.",
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_BODY_MODEL_FOLDER,
        help="Folder containing SMPL body model files (e.g., transfer_data/body_models).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        help="SMPL gender to use for the template faces.",
    )
    parser.add_argument(
        "--model-ext",
        type=str,
        default="npz",
        help="Extension of the SMPL model files (npz or pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = export_npz_to_obj(
        dataset_root=args.dataset_root,
        obj_output=args.obj_output,
        model_folder=args.model_folder,
        gender=args.gender,
        model_ext=args.model_ext,
    )
    print(f"Wrote {len(written)} meshes to {args.obj_output}")


if __name__ == "__main__":
    main()
