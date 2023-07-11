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
    # data_path = os.path.join("../../pickles", 'training_set_with_bogus.pkl')
    data_path = "../../pickles/converted_data.pkl"
    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "bn_test",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
    }
    trainer = Trainer(params)

    # # no bn
    # trainer.train_model_n_times(DeepHiTS4ClassModelNanNormStamp, params,
    #                             train_times=10,
    #                             model_name='DHNan0NormStampWBogus')
    #
    # # bn after layers
    # params.update({param_keys.BATCHNORM_CONV: constants.BN,
    #                param_keys.BATCHNORM_FC: None})
    # trainer.train_model_n_times(DeepHiTS4ClassModelNanNormStamp, params,
    #                             train_times=10,
    #                             model_name='DHNan0NormStampWBogusConvBN')
    #
    # params.update({param_keys.BATCHNORM_CONV: constants.BN,
    #                param_keys.BATCHNORM_FC: constants.BN})
    # trainer.train_model_n_times(
    #     DeepHiTS4ClassModelNanNormStamp, params, train_times=10,
    #     model_name='DHNan0NormStampWBogusConvBNFcBN')
    #
    # params.update({param_keys.BATCHNORM_CONV: constants.BN_RENORM,
    #                param_keys.BATCHNORM_FC: None})
    # trainer.train_model_n_times(
    #     DeepHiTS4ClassModelNanNormStamp, params, train_times=10,
    #     model_name='DHNan0NormStampWBogusConvBnRenorm')
    #
    # params.update({param_keys.BATCHNORM_CONV: constants.BN_RENORM,
    #                param_keys.BATCHNORM_FC: constants.BN_RENORM})
    # trainer.train_model_n_times(
    #     DeepHiTS4ClassModelNanNormStamp, params, train_times=10,
    #     model_name='DHNan0NormStampWBogusConvBnRenormFcBnRenorm')

    # bn input and after layers
    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBNNan0NormStampWBogusConvBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN, param_keys.BATCHNORM_FC: constants.BN}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBNNan0NormStampWBogusConvBNFcBN",
    )

    params.update(
        {param_keys.BATCHNORM_CONV: constants.BN_RENORM, param_keys.BATCHNORM_FC: None}
    )
    trainer.train_model_n_times(
        DeepHiTSBNNanNormStampModel,
        params,
        train_times=10,
        model_name="DHBNNan0NormStampWBogusConvBnRenorm",
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
        model_name="DHBNNan0NormStampWBogusConvBnRenormFcBnRenorm",
    )

    trainer.print_all_accuracies()
