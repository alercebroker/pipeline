"""Module that defines the parameters to configure the model and their
default values."""

import os
import sys

from . import constants
from . import general_keys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

# Input params
"""
validation_random_seed: (int) Random seed to be used to split the validation
    data from the training data.
"""
DATA_PATH_TRAIN = "data_path"
CHECKPOINT_PATH_TO_START_FINETUNE = "checkpoint_path_to_star_fine_tune"
DATA_PATH_TRANSFER_LEARNING = "data_path_4_transfer_learning"
DATA_PATH_TEST = "data_path_to_test_sets"
DATA_PATH_VALIDATION = "data_path_to_validate_sets"
DATA_PATHS_DICT = "data_paths_dict"
DATA_PATH_ELASTIC = "data_path_elastic"
BATCH_SIZE = "batch_size"
NUMBER_OF_CLASSES = "number_of_classes"
N_INPUT_CHANNELS = "n_input_channels"
INPUT_TIME_SERIES_SIZE = "input_time_series_size"
INPUT_IMAGE_SIZE = "input_image_size"
SHUFFLE_BUFFER_SIZE = "shuffle_buffer_size"
PREFETCH_BUFFER_SIZE = "prefetch_buffer_size"
VALIDATION_RANDOM_SEED = "validation_random_seed"
TEST_RANDOM_SEED = "test_random_seed"
DATA_TYPES_DICT = "data_types_dict"

BATCH_SIZE_ORIGINAL = "batch_size_original"
BATCH_SIZE_ELASTIC = "batch_size_elastic"

CHANNELS_TO_USE = "channels_to_use"

SHIFT_MAX = "shift_max"
SHIFT_NUMBER = "shift_number"
ANGLE_STEP = "angle_step"

# Model params
"""
batchnorm_conv: ({None, BN, BN_RENORM}) Type of batchnorm to be used in the
    convolutions. If None, no batchnorm is applied.
drop_rate: ({None, int}) Dropout rate to be applied. If None, no dropout is
    applied.
"""
BATCHNORM_CONV = "batchnorm_conv"
BATCHNORM_FC = "batchnorm_fc"
DROP_RATE = "drop_rate"
INITIAL_N_FILTERS = "initial_conv_filters"
KERNEL_SIZE = "kernel_size"
POOL_SIZE = "pool_size"
BATCHNORM_FEATURES_FC = "batchnorm_features_fc"

# Loss params
"""
type_loss: ({CROSS_ENTROPY_LOSS, DICE_LOSS}) Type of loss to be minimized.
"""
# TYPE_LOSS = 'type_loss'

# Optimizer params
"""
learning_rate: (float) initial learning rate value
type_optimizer: ({SGD_OPTIMIZER, MOMENTUM_SGD_OPTIMIZER, ADAM_OPTIMIZER}) Type
    of optimizer to be used.
momentum: (float) Momentum to be used with MOMENTUM_SGD_OPTIMIZER. 
    If other optimizer is used, then this parameter has no effect.
use_nesterov_momentum: (bool) If true, and if a momentum optimizer is used, 
    Nesterov momentum is applied.
"""
LEARNING_RATE = "learning_rate"
BETA1 = "beta1"
BETA2 = "beta2"

LAMBDA = "lambda"  # Gradient penalty lambda hyperparameter
# TYPE_OPTIMIZER = 'type_optimizer'
# MOMENTUM = 'momentum'
# USE_NESTEROV_MOMENTUM = 'use_nesterov_momentum'

# Training params
"""
print_every: (int) How many iterations to wait before printing stats.
train_iterations_horizon: (int) How many iterations to train.
iterations_to_update_learning_rate: (int) How many iterations to wait before
    decreasing the learning rate
iterations_to_validate: (int) How many iterations to wait before validating.
criterion_percentage: (float between 0 and 1) Percentage of change wrt the last 
    best loss, to verify that the loss is still improving and we have to
    increment the training iterations.
train_horizon_increment: (int) How many iterations to be added if criterion
    percentage is met.
"""
PRINT_EVERY = "print_every"
TRAIN_ITERATIONS_HORIZON = "train_iterations_horizon"
ITERATIONS_TO_UPDATE_LEARNING_RATE = "iterations_to_update_learning_rate"
ITERATIONS_TO_VALIDATE = "iterations_to_validate"
TRAIN_HORIZON_INCREMENT = "train_horizon_increment"
CRITERION_PERCENTAGE = "criterion_percentage"

NOISE_DIM = "noise_dim"
DISC_TRAINS = "disc_trains"
GEN_TRAINS = "gen_trains"

TEST_SIZE = "test_size"
VAL_SIZE = "val_size"

RESULTS_FOLDER_NAME = "results_folder_name"

WAIT_FIRST_EPOCH = "wait_first_epoch"

NANS_TO = "nans_to"
CROP_SIZE = "crop_size"
BOGUS_LABEL_VALUE = "bogus_label_value"
CONVERTED_DATA_SAVEPATH = "converted_data_savepath"

VALIDATION_MONITOR = "validation_monitor"
VALIDATION_MODE = "validation_mode"

ENTROPY_REG_BETA = "entropy_regularizer_beta"
INPUT_DATA_PREPROCESSOR = "input_preprocessing_pipeline"
FEATURES_NAMES_LIST = "features_names_list"
# Default parameters dictionary


# AE visualizer params
BOTTLE_NECK_DIM = "bottle_neck_dim"

FEATURES_CLIPPING_DICT = "features_clipping_dict"
"""
Usage:
    Import the default parameters dictionary as:
    
        from parameters import param_keys
        my_dict = param_keys.default_params
    
    Then, if you want to overwrite some values that are inside a dictionary
    named "my_custom_parameters", you can simply do:
    
        my_dict.update(my_custom_parameters)  # Overwrite defaults
        
"""
default_params = {
    DATA_PATH_TRAIN: None,
    DATA_PATH_TRANSFER_LEARNING: None,
    DATA_PATH_TEST: None,
    DATA_PATH_VALIDATION: None,
    # DATA_PATH_ELASTIC: None,
    CHECKPOINT_PATH_TO_START_FINETUNE: None,
    BATCH_SIZE: 32,
    INITIAL_N_FILTERS: 64,
    NOISE_DIM: 128,
    LAMBDA: 10,
    DISC_TRAINS: 5,
    CHANNELS_TO_USE: 1,
    WAIT_FIRST_EPOCH: False,
    # BATCH_SIZE_ORIGINAL: 4,
    # BATCH_SIZE_ELASTIC: 4,
    # SHIFT_MAX: 0.15,
    # SHIFT_NUMBER: 4,
    # ANGLE_STEP: 20,
    NUMBER_OF_CLASSES: 2,
    N_INPUT_CHANNELS: 3,
    INPUT_IMAGE_SIZE: 63,
    SHUFFLE_BUFFER_SIZE: 100000,
    PREFETCH_BUFFER_SIZE: 100,
    VALIDATION_RANDOM_SEED: 42,
    TEST_RANDOM_SEED: 42,
    DROP_RATE: 0.5,
    # BATCHNORM_CONV: constants.BN,
    # BATCHNORM_FC: constants.BN,
    BATCHNORM_FC: None,
    BATCHNORM_CONV: None,
    BATCHNORM_FEATURES_FC: False,
    KERNEL_SIZE: 3,
    POOL_SIZE: 2,
    # INITIAL_CONV_FILTERS: 64,
    LEARNING_RATE: 1e-4,
    BETA1: 0.5,
    BETA2: 0.9,
    PRINT_EVERY: 10,
    TRAIN_ITERATIONS_HORIZON: 10000,
    ITERATIONS_TO_UPDATE_LEARNING_RATE: 0,
    ITERATIONS_TO_VALIDATE: 10,
    CRITERION_PERCENTAGE: 0.0,
    TEST_SIZE: 200,
    VAL_SIZE: 200,
    NANS_TO: 0,
    RESULTS_FOLDER_NAME: "",
    BOGUS_LABEL_VALUE: None,
    CONVERTED_DATA_SAVEPATH: os.path.join(
        PROJECT_PATH, "..", "pickles", "converted_data.pkl"
    ),
    VALIDATION_MONITOR: general_keys.LOSS,
    VALIDATION_MODE: general_keys.MIN,
    CROP_SIZE: None,
    ENTROPY_REG_BETA: None,
    INPUT_DATA_PREPROCESSOR: None,
    FEATURES_NAMES_LIST: [
        "sgscore1",
        "distpsnr1",
        "sgscore2",
        "distpsnr2",
        "sgscore3",
        "distpsnr3",
        "isdiffpos",
    ],
    FEATURES_CLIPPING_DICT: {},
    DATA_TYPES_DICT: {
        general_keys.TRAIN: general_keys.REAL,
        general_keys.VALIDATION: general_keys.REAL,
        general_keys.TEST: general_keys.REAL,
    },
}
# Now set default values that depend on other params
default_params.update(
    {
        TRAIN_HORIZON_INCREMENT: default_params[TRAIN_ITERATIONS_HORIZON],
        # DATA_PATH_VALIDATION: default_params[DATA_PATH_TEST]
    }
)


def update_paths_dict(params):
    params.update(
        {
            DATA_PATHS_DICT: {
                general_keys.TRAIN: params[DATA_PATH_TRAIN],
                general_keys.VALIDATION: params[DATA_PATH_VALIDATION],
                general_keys.TEST: params[DATA_PATH_TEST],
            }
        }
    )
