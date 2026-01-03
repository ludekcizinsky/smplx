#!/bin/bash
# set -e

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Variables that need to be set
SEQ_ROOT=$1 # eg. "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19"

# Derived paths
# - where to load the npz smplx from
SMPLX_DIR=$SEQ_ROOT/smplx
# - where to output the obj meshes
OBJ_OUT=$SEQ_ROOT/misc/smplx_meshes
mkdir -p $OBJ_OUT
# - where to output the smplx batch fitting pkls
FITTING_OUT_DIR=$SEQ_ROOT/misc/smplx2smpl_out
mkdir -p $FITTING_OUT_DIR
# - where to save the resulting smpl npz 
SMPL_DIR=$SEQ_ROOT/smpl
mkdir -p $SMPL_DIR

# Configuration for where the smplx repo lives
ROOT="/home/cizinsky/master-thesis/submodules/smplx"
CONFIG="$ROOT/config_files/custom_smplx2smpl.yaml" 
cd "$ROOT"

# Conversion steps from smplx to smpl:
# echo "Exporting SMPL-X npz frames to obj meshes..."
# conda activate thesis
# python3 "$ROOT/tools/body_model_to_mesh.py" \
  # --dataset-root "$SMPLX_DIR" \
  # --obj-output "$OBJ_OUT" \
  # --model-folder "/home/cizinsky/body_models" \
  # --model-type smplx

# echo "Running SMPLX -> SMPL fitting..."
# conda activate smplx
# python -m transfer_model --exp-cfg "$CONFIG" --exp-opts batch_size=1000 use_cuda=True \
  # datasets.mesh_folder.data_folder="$OBJ_OUT" \
  # output_folder="$FITTING_OUT_DIR"

echo "Packing SMPL pkls into per-frame npz under smpl/..."
conda activate smplx
python "$ROOT/tools/pack_smplx_npz.py" \
  --pkl-folder "$FITTING_OUT_DIR" \
  --out-folder "$SMPL_DIR" \
  --model-type smpl
