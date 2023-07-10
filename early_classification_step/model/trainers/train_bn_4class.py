import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_4class_nans_norm_stamp_model import (
    DeepHiTS4ClassModelNanNormStamp,
)
from models.classifiers.deepHitsBN_nans_norm_stamp_model import (
    DeepHiTSBNNanNormStampModel,
)
from trainers.base_trainer import Trainer
from parameters import param_keys, constants

if __name__ == "__main__":
    data_path = os.path.join("../../pickles", "corrected_oids_alerts.pkl")
    # data_path = "../../pickles/converted_data.pkl"
    params = {
        param_keys.RESULTS_FOLDER_NAME: "bn_4classes",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: 200,
        param_keys.VAL_SIZE: 200,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: 4,
    }
    trainer = Trainer(params)

    # no bn
    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStamp",
    )

    # bn after layers
    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStampConvBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: constants.BN}
    )
    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStampConvBNFcBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN_RENORM, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStampConvBnRenorm",
    )

    params.update(
        {
            param_keys.BATCHNORM_CONV: constants.BN_RENORM,
            param_keys.BATCHNORM_FC: constants.BN_RENORM,
        }
    )
    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStampConvBnRenormFcBnRenorm",
    )

    # bn input and after layers
    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBN4ClassNan0NormStampConvBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: constants.BN}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBN4ClassNan0NormStampConvBNFcBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN_RENORM, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBN4ClassNan0NormStampConvBnRenorm",
    )

    params.update(
        {
            param_keys.BATCHNORM_CONV: constants.BN_RENORM,
            param_keys.BATCHNORM_FC: constants.BN_RENORM,
        }
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBN4ClassNan0NormStampConvBnRenormFcBnRenorm",
    )

    trainer.print_all_accuracies()
