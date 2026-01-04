#!/bin/bash
# set -e

# init conda
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Variables that need to be set
SEQ_ROOT=$1 # e.g. "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19"
FROM_TYPE=$2 # e.g. smplx
TO_TYPE=$3   # e.g. smpl

# Derived paths
# - where to load the npz smplx from
SRC_DIR=$SEQ_ROOT/$FROM_TYPE
# - where to output the obj meshes
OBJ_OUT=$SEQ_ROOT/misc/${FROM_TYPE}_meshes
mkdir -p $OBJ_OUT
# - where to output the smplx batch fitting pkls
FITTING_OUT_DIR=$SEQ_ROOT/misc/${FROM_TYPE}2${TO_TYPE}_out
mkdir -p $FITTING_OUT_DIR
# - where to save the resulting smpl npz 
TGT_DIR=$SEQ_ROOT/$TO_TYPE
mkdir -p $TGT_DIR

# Configuration for where the smplx repo lives
ROOT=/home/cizinsky/master-thesis/submodules/smplx
CONFIG=$ROOT/config_files/custom_${FROM_TYPE}2${TO_TYPE}.yaml
cd $ROOT

# Only use mask ids when converting from SMPL -> SMPLX.
MASK_IDS_FNAME=""
if [ "$FROM_TYPE" = "smpl" ]; then
  MASK_IDS_FNAME="$ROOT/transfer_data/smplx_mask_ids.npy"
fi
MASK_IDS_OPT=""
if [ -n "$MASK_IDS_FNAME" ]; then
  MASK_IDS_OPT="mask_ids_fname=$MASK_IDS_FNAME"
fi

# Conversion steps from smplx to smpl:
echo "Exporting $FROM_TYPE npz frames to obj meshes..."
conda activate thesis
python3 $ROOT/tools/body_model_to_mesh.py \
  --dataset-root $SRC_DIR \
  --obj-output $OBJ_OUT \
  --model-folder "/home/cizinsky/body_models" \
  --model-type $FROM_TYPE

echo "Running $FROM_TYPE -> $TO_TYPE fitting..."
conda activate smplx
python -m transfer_model --exp-cfg $CONFIG --exp-opts batch_size=1000 use_cuda=True \
  datasets.mesh_folder.data_folder=$OBJ_OUT \
  output_folder=$FITTING_OUT_DIR \
  deformation_transfer_path=$ROOT/transfer_data/${FROM_TYPE}2${TO_TYPE}_deftrafo_setup.pkl \
  $MASK_IDS_OPT

echo "Packing $TO_TYPE pkls into per-frame npz under $TO_TYPE/..."
conda activate smplx
python $ROOT/tools/pack_to_npz.py \
  --pkl-folder $FITTING_OUT_DIR \
  --out-folder $TGT_DIR \
  --model-type $TO_TYPE
