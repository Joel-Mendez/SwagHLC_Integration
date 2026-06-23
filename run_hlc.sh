#!/bin/bash
cd "$(dirname "$0")"
export RRD_ROOT=/home/swag/Desktop/SWAG_Project/SampleData/RRD
export SWAG_MODELS_ROOT=/home/swag/Desktop/SWAG_Project/SampleData/SampleIntegrationData/SampleIntegrationData/SWAG-TrainedModels/Integration_models
.venv/bin/python -m swag_hlc.app --config configs/integration_ros2.yaml
