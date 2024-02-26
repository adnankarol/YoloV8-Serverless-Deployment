#!/bin/bash

ENV_NAME="yolo"
CONDA_ACTIVATE_DIR=`which activate`

echo "setting $ENV_NAME environment"

# Create Conda Virtual Environment
conda create --name $ENV_NAME python=3.10 --y

# Activate Virtual Environment
source $CONDA_ACTIVATE_DIR $ENV_NAME

echo "python version is:"
python --version

echo "upgrading pip..."
pip install --upgrade pip

echo "conda env created and activated successfully"

# Install Required Packages
echo "Installing requirements.txt packages..."
pip install -r requirements.txt

echo "packages installed successfully"

conda deactivate
