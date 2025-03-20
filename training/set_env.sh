#!/bin/bash

# Set environment variables
export TEST_ANOMALY_QUANTILES_PATH=https://alerce-models.s3.amazonaws.com/anomaly/1.0.2/quantiles/
export TEST_ANOMALY_MODEL_PATH=https://alerce-models.s3.amazonaws.com/anomaly/1.0.2/models/
export TEST_MBAPPE_QUANTILES_PATH=https://alerce-models.s3.amazonaws.com/mbappe/0.3.6/quantiles
export TEST_MBAPPE_CONFIG_PATH=https://alerce-models.s3.amazonaws.com/mbappe/0.3.6/configs
export TEST_MBAPPE_MODEL_PATH=https://alerce-models.s3.amazonaws.com/mbappe/0.3.6/model.ckpt
export TEST_SQUIDWARD_MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/1.1.5/hierarchical_random_forest_model.pkl
export TEST_STAMP_FULL_MODEL_PATH=https://alerce-models.s3.amazonaws.com/stamp_full/1.0.0/model.keras
export COMPOSE=v2

# Confirm variables are set
echo "Environment variables set successfully."