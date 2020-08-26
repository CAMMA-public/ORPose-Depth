#!/usr/bin/env bash
# coding: utf-8
# Author: Vinkle Srivastav (srivastav@unistra.fr)
# -----
# Copyright 2018 - (C) Copyright University of Strasbourg, All Rights Reserved.
# '''

# activate the conda environment
source $(conda info --base)/bin/activate
conda activate depthpose_env
cd ..


echo "------------------------------------- Evaluation starts -------------------------------------------------"
# --use-cpu  flag to run the evaluation on the cpu

# To run the evaluation for DepthPose_64x48 model
CONFIG_FILE=experiments/mvor/DepthPose_64x48.yaml
python tools/eval_mvor.py --config_file ${CONFIG_FILE}                    


# To run the evaluation for DepthPose_80x60 model
# CONFIG_FILE=experiments/mvor/DepthPose_80x60.yaml
python tools/eval_mvor.py --config_file ${CONFIG_FILE}                      
echo "------------------------------------- Evaluation ends ----------------------------------------------------"
