#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import smplx
import torch
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


def _create_model(
    model_folder: Path,
    model_type: str,
    gender: str,
    ext: str,
    num_expression_coeffs: int | None,
    use_pca: bool | None,
    num_pca_comps: int | None,
) -> smplx.body_models.SMPL:
    kwargs = {}
    if num_expression_coeffs is not None:
        kwargs["num_expression_coeffs"] = num_expression_coeffs
    if use_pca is not None:
        kwargs["use_pca"] = use_pca
    if num_pca_comps is not None:
        kwargs["num_pca_comps"] = num_pca_comps
    return smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        ext=ext,
        **kwargs,
    )


def _infer_num_expression_coeffs(data: np.lib.npyio.NpzFile) -> int | None:
    if "expression" not in data:
        return None
    expr = data["expression"]
    if expr.ndim == 2:
        return int(expr.shape[1])
    if expr.ndim == 1:
        return int(expr.shape[0])
    return None


def _hand_pose_format(hand_pose: np.ndarray) -> tuple[str, int]:
    hand_pose = np.asarray(hand_pose)
    if hand_pose.ndim == 1:
        hand_pose = hand_pose[None, :]
    if hand_pose.ndim == 3:
        if hand_pose.shape[-1] != 3:
            raise ValueError(f"Unexpected hand pose shape {hand_pose.shape}")
        return "axis-angle", hand_pose.shape[1] * hand_pose.shape[2]
    if hand_pose.ndim == 2:
        if hand_pose.shape[1] == 45:
            return "axis-angle", hand_pose.shape[1]
        return "pca", hand_pose.shape[1]
    raise ValueError(f"Unexpected hand pose shape {hand_pose.shape}")


def _infer_hand_pose_config(
    data: np.lib.npyio.NpzFile,
) -> tuple[bool | None, int | None]:
    hand_keys = [key for key in ("lhand_pose", "rhand_pose") if key in data]
    if not hand_keys:
        return None, None
    formats = []
    for key in hand_keys:
        formats.append(_hand_pose_format(data[key]))
    if any(fmt == "axis-angle" for fmt, _ in formats):
        return False, None
    pca_dims = {dim for _, dim in formats}
    if len(pca_dims) != 1:
        raise ValueError(f"Hand pose PCA dims mismatch: {sorted(pca_dims)}")
    return True, pca_dims.pop()


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim == 3:
        if pose.shape[-1] != 3:
            raise ValueError(f"Unexpected pose shape {pose.shape}")
        return pose.reshape(pose.shape[0], -1)
    if pose.ndim == 2:
        return pose
    raise ValueError(f"Unexpected pose shape {pose.shape}")


def _normalize_hand_pose(hand_pose: np.ndarray, use_pca: bool | None) -> np.ndarray:
    hand_pose = np.asarray(hand_pose)
    if hand_pose.ndim == 1:
        hand_pose = hand_pose[None, :]
    if use_pca:
        if hand_pose.ndim != 2:
            raise ValueError(f"Unexpected PCA hand pose shape {hand_pose.shape}")
        return hand_pose
    if hand_pose.ndim == 3:
        if hand_pose.shape[-1] != 3:
            raise ValueError(f"Unexpected axis-angle hand pose shape {hand_pose.shape}")
        return hand_pose.reshape(hand_pose.shape[0], -1)
    if hand_pose.ndim == 2:
        if use_pca is False and hand_pose.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected axis-angle hand pose shape {hand_pose.shape}")
        return hand_pose
    raise ValueError(f"Unexpected hand pose shape {hand_pose.shape}")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector)
    if vector.ndim == 1:
        return vector[None, :]
    if vector.ndim == 2:
        return vector
    raise ValueError(f"Unexpected vector shape {vector.shape}")


def _npz_to_vertices(
    data: np.lib.npyio.NpzFile,
    model: smplx.body_models.SMPL | None,
    model_type: str,
    hand_use_pca: bool | None,
) -> np.ndarray:
    if "verts" in data:
        verts = data["verts"]
        if verts.ndim != 3:
            raise ValueError(f"Unexpected verts shape {verts.shape}")
        return verts

    if model is None:
        raise RuntimeError("SMPL/SMPL-X model is required to generate vertices.")

    required = {"betas", "root_pose", "body_pose", "trans"}
    available = set(data.keys())
    missing = required - available
    if missing:
        raise KeyError(f"Missing keys {sorted(missing)}; available keys: {sorted(available)}")

    device = next(model.parameters()).device
    to_tensor = lambda value: torch.as_tensor(
        value, dtype=torch.float32, device=device
    )

    params = {
        "betas": to_tensor(_normalize_vector(data["betas"])),
        "global_orient": to_tensor(_normalize_pose(data["root_pose"])),
        "body_pose": to_tensor(_normalize_pose(data["body_pose"])),
        "transl": to_tensor(_normalize_vector(data["trans"])),
    }

    if model_type == "smplx":
        if "jaw_pose" in data:
            params["jaw_pose"] = to_tensor(_normalize_pose(data["jaw_pose"]))
        if "leye_pose" in data:
            params["leye_pose"] = to_tensor(_normalize_pose(data["leye_pose"]))
        if "reye_pose" in data:
            params["reye_pose"] = to_tensor(_normalize_pose(data["reye_pose"]))
        if "lhand_pose" in data:
            params["left_hand_pose"] = to_tensor(
                _normalize_hand_pose(data["lhand_pose"], hand_use_pca)
            )
        if "rhand_pose" in data:
            params["right_hand_pose"] = to_tensor(
                _normalize_hand_pose(data["rhand_pose"], hand_use_pca)
            )
        if "expression" in data:
            params["expression"] = to_tensor(_normalize_vector(data["expression"]))

    with torch.no_grad():
        output = model(**params, return_verts=True)
    return output.vertices.detach().cpu().numpy()


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

    # Lazily create the model when needed (when no verts are in the npz)
    model = None
    model_expr_coeffs = None
    model_use_pca = None
    model_num_pca_comps = None
    out_model_type = "smpl" if model_type == "smplx" else model_type
    placeholder_root = dataset_root.parent / out_model_type
    placeholder_root.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(dataset_root.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {dataset_root}")

    written = []
    for npz_path in tqdm(npz_files, desc="Exporting SMPL meshes"):
        with np.load(npz_path, allow_pickle=True) as data:
            # Handle empty npz files by writing empty placeholders
            if not data.keys():
                print(f"Skipping empty npz: {npz_path}, and writing empty placeholder to {placeholder_root}")
                np.savez(placeholder_root / npz_path.name)
                continue

            # Lazily create the model if needed
            if "verts" not in data and model is None:
                model_expr_coeffs = _infer_num_expression_coeffs(data)
                model_use_pca, model_num_pca_comps = _infer_hand_pose_config(data)
                model = _create_model(
                    model_folder=model_folder,
                    model_type=model_type,
                    gender=gender,
                    ext=model_ext,
                    num_expression_coeffs=model_expr_coeffs,
                    use_pca=model_use_pca,
                    num_pca_comps=model_num_pca_comps,
                )

            # Convert npz to vertices
            verts = _npz_to_vertices(data, model, model_type, model_use_pca)

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
