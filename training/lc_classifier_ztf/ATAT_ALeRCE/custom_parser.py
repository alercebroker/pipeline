import argparse
import yaml


def parse_model_args(arg_dict=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type_general", type=str, default="lc")
    parser.add_argument("--experiment_name_general", type=str, default="lc_tm")
    parser.add_argument("--name_dataset_general", type=str, default="ztf")
    parser.add_argument(
        "--data_root_general", type=str, default="data/final/ZTF_ff/LC_MD_FEAT_v2"
    )

    ## Lightcurves
    parser.add_argument("--use_lightcurves_general", action="store_true", default=False)
    parser.add_argument(
        "--use_lightcurves_err_general", action="store_true", default=False
    )
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=192)
    parser.add_argument("--embedding_size_sub", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_encoders", type=int, default=3)
    parser.add_argument(
        "--Tmax", type=float, default=1500.0
    )  # 2500 --> para ZTF, no lo he probado
    parser.add_argument("--num_harmonics", type=int, default=64)
    parser.add_argument("--pe_type", type=str, default="tm")

    ## Metadata and Features
    parser.add_argument("--use_metadata_general", action="store_true", default=False)
    parser.add_argument("--use_features_general", action="store_true", default=False)
    parser.add_argument("--embedding_size_tab", type=int, default=128)
    parser.add_argument("--embedding_size_tab_sub", type=int, default=256)
    parser.add_argument("--num_heads_tab", type=int, default=4)
    parser.add_argument("--num_encoders_tab", type=int, default=3)

    # CNN PARAMS
    parser.add_argument("--encoder_type", type=str, default="Linear")
    parser.add_argument("--encoder_type_tab", type=str, default="Linear")
    parser.add_argument("--max_pool_kernel", type=int, default=5)
    parser.add_argument("--cnn_kernel", type=int, default=5)

    # TRAINING PARAMS
    parser.add_argument("--batch_size_general", type=int, default=256)
    parser.add_argument("--num_epochs_general", type=int, default=10000)
    parser.add_argument("--patience_general", type=int, default=40)
    parser.add_argument("--lr_general", type=float, default=1e-4)
    parser.add_argument(
        "--use_cosine_decay_general", action="store_true", default=False
    )
    parser.add_argument(
        "--use_gradient_clipping_general", action="store_true", default=False
    )

    parser.add_argument(
        "--use_mask_detection_general", action="store_true", default=False
    )
    parser.add_argument(
        "--use_time_nondetection_general", action="store_true", default=False
    )

    # AUGMENTATIONS
    parser.add_argument(
        "--force_online_opt_general", action="store_true", default=False
    )  # No revisado por mi
    parser.add_argument("--online_opt_tt_general", action="store_true", default=False)

    # ABLATION
    parser.add_argument("--use_QT_general", action="store_true", default=False)

    # LOAD MODEL
    parser.add_argument(
        "--load_pretrained_model_general", action="store_true", default=False
    )
    parser.add_argument("--src_checkpoint_general", type=str, default=".")

    parser.add_argument(
        "--use_augmented_dataset_general", action="store_true", default=False
    )
    parser.add_argument("--change_clf_general", action="store_true", default=False)

    args = parser.parse_args(None if arg_dict is None else [])
    required_args = ["Tmax"]

    for key in required_args:
        if getattr(args, key) is None:
            parser.error("Missing argument {}".format(key))

    return args


def handler_parser(
    parser_dict, extra_args_general=None, extra_args_lc=None, extra_args_tab=None
):

    with open(
        "./{}/dict_info.yaml".format(parser_dict["data_root_general"]), "r"
    ) as yaml_file:
        dict_info = yaml.safe_load(yaml_file)

    parser_dict.update(
        {
            "num_classes_general": len(dict_info["classes_to_use"]),
            "num_bands": len(dict_info["bands_to_use"]),
            "length_size_tab": 0,
            "list_time_to_eval_tab": None,
        }
    )

    if parser_dict["use_lightcurves_general"]:
        parser_dict["input_size"] = 1

    if parser_dict["use_lightcurves_err_general"]:
        parser_dict["input_size"] = 2

    if parser_dict["use_metadata_general"]:
        parser_dict["length_size_tab"] += len(dict_info["md_cols"])

    if parser_dict["use_features_general"]:
        parser_dict["length_size_tab"] += len(dict_info["feat_cols"])
        parser_dict["list_time_to_eval_general"] = dict_info["list_time_to_eval"]

    output = {}
    output["lc"] = {}
    output["ft"] = {}
    output["general"] = {}

    for key in parser_dict.keys():
        # settings for tabular transformer
        if "tab" in key:
            output["ft"][key.replace("_tab", "")] = parser_dict[key]
        # general settings
        elif "general" in key:
            output["general"][key.replace("_general", "")] = parser_dict[key]
        # settings for lightcurve trasnformer
        else:
            output["lc"][key] = parser_dict[key]

    if extra_args_general is not None:
        output["general"].update(extra_args_general)
    if extra_args_lc is not None:
        output["lc"].update(extra_args_lc)
    if extra_args_tab is not None:
        output["ft"].update(extra_args_tab)

    return output
