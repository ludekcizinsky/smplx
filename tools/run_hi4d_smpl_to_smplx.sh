#!/bin/bash
# set -e

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

ROOT="/home/cizinsky/smplx"
SEQ_ROOT="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/fight00"
SMPL_DIR="$SEQ_ROOT/smpl"
SMPLX_BATCH_DIR="$SEQ_ROOT/fitted_smplx_batch"
SMPLX_DIR="$SEQ_ROOT/smplx"
OBJ_OUT="$SEQ_ROOT/smpl_to_smplx_meshes"
CONFIG="$ROOT/config_files/hi4d_smpl2smplx.yaml"

cd "$ROOT"

# echo "Exporting SMPL npz frames to obj meshes..."
# conda activate thesis
# python3 "$ROOT/tools/convert_hi4d_smpl_to_smplx.py" \
  # --dataset-root "$SMPL_DIR" \
  # --obj-output "$OBJ_OUT" \
  # --model-folder "/home/cizinsky/body_models"

# echo "Running SMPL -> SMPL-X fitting..."
# conda activate smplx
# python -m transfer_model --exp-cfg "$CONFIG" --exp-opts batch_size=50 use_cuda=True

echo "Packing SMPL-X pkls into per-frame npz under smplx/..."
conda activate smplx
python "$ROOT/tools/pack_smplx_npz.py" \
  --pkl-folder "$SMPLX_BATCH_DIR" \
  --out-folder "$SMPLX_DIR"
