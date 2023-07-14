"""Modules that defines useful constants for the project."""

# Database split
TRAIN_SUBSET = "train"
VAL_SUBSET = "val"
TEST_SUBSET = "test"

# Model names
GENERATOR = "Generator"
DISCRIMINATOR = "Discriminator"
DECODER = "decoder"

# Type of padding
PAD_SAME = "same"
PAD_VALID = "valid"

# Type of batch normalization
BN = "bn"
BN_RENORM = "bn_renorm"

# Type of pooling
MAXPOOL = "maxpool"
AVGPOOL = "avgpool"

# Type of connections management for initializers
FAN_IN = "FAN_IN"
FAN_OUT = "FAN_OUT"
FAN_AVG = "FAN_AVG"

# Types of loss
CROSS_ENTROPY_LOSS = "cross_entropy_loss"
DICE_LOSS = "dice_loss"

# Types of optimizer
ADAM_OPTIMIZER = "adam_optimizer"
SGD_OPTIMIZER = "sgd_optimizer"
MOMENTUM_SGD_OPTIMIZER = "momentum_sgd_optimizer"

# Error message, parameter not in valid list
ERROR_INVALID = "Expected %s for %s, but %s was provided."
# Error message, metric value is nan
ERROR_NAN_METRIC = "Metric %s has NaN values; munsn't be calculated with batch means"
# Error message, when data_frame don't contain all wanted features
ERROR_DATAFRAME_FEATURES = "Dataframe lacks the following features %s"
