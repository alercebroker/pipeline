import os
import sys
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import pprint
import pickle
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


def get_weight_dict(chkp_path, save_folder=None):
    weights_dict = {}
    reader = pywrap_tensorflow.NewCheckpointReader(chkp_path)
    name_shape_dict = reader.get_variable_to_shape_map()
    for tensor_name in name_shape_dict.keys():
        if (
            len(name_shape_dict[tensor_name]) == 0
            or "Adam" in tensor_name
            or "opaque_kernel" in tensor_name
        ):
            continue
        splited_name = tensor_name.split("/")
        layer_name = splited_name[1]

        if "conv" in layer_name:
            layer_text_splitted = layer_name.split("_")
            layer_instance_number = layer_text_splitted[1]
            tensor = reader.get_tensor(tensor_name)
            if "bias" in splited_name:
                np.save(
                    os.path.join(
                        save_folder, "CNN%i-B.npy" % int(layer_instance_number)
                    ),
                    tensor,
                )
            if "kernel" in splited_name:
                np.save(
                    os.path.join(
                        save_folder, "CNN%i-W.npy" % int(layer_instance_number)
                    ),
                    tensor,
                )
            if layer_name not in weights_dict:
                weights_dict[layer_name] = {}
            weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(tensor_name)

        if "dense" in layer_name:
            layer_text_splitted = layer_name.split("_")
            layer_instance_number = layer_text_splitted[1]
            tensor = reader.get_tensor(tensor_name)
            if "bias" in splited_name:
                np.save(
                    os.path.join(
                        save_folder, "FC%i-B.npy" % int(layer_instance_number)
                    ),
                    tensor,
                )
            if "kernel" in splited_name:
                np.save(
                    os.path.join(
                        save_folder, "FC%i-W.npy" % int(layer_instance_number)
                    ),
                    tensor,
                )
            if layer_name not in weights_dict:
                weights_dict[layer_name] = {}
            weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(tensor_name)

        if "output_logits" in layer_name:
            layer_text_splitted = layer_name.split("_")
            layer_instance_number = layer_text_splitted[1]
            tensor = reader.get_tensor(tensor_name)
            if "bias" in layer_name:
                np.save("FC3-B.npy", tensor)
            if "kernel" in splited_name:
                np.save(os.path.join(save_folder, "FC3-W.npy"), tensor)
            if layer_name not in weights_dict:
                weights_dict[layer_name] = {}
            weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(tensor_name)

        print("layer_names: ", np.sort(list(weights_dict.keys())))

    if np.unique(list(weights_dict.keys())).shape[0] != len(list(weights_dict.keys())):
        raise ValueError(
            "dimensions mismatch, there is a layer_name appearing more than once"
        )
    if save_folder is not None:
        with open(
            os.path.join(save_folder, "checkpoint_weightsas_dict_of_numpys.pickle"),
            "wb",
        ) as handle:
            pickle.dump(weights_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return weights_dict


if __name__ == "__main__":
    chkp = os.path.join(project_root, "results/best_model_so_far/checkpoints", "model")
    reader = pywrap_tensorflow.NewCheckpointReader(chkp)
    tensor_names = reader.debug_string().decode("utf-8")
    tensor_names_list = tensor_names.split("\n")
    # a = reader.get_tensor('model_v10/multi_layer_blstm/lstm_1/cudnn_blstm/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/cudnn_compatible_lstm_cell/bias')
    # print(a.shape)
    print(reader.get_variable_to_shape_map())

    weight_folder = os.path.join(
        project_root, "scripts", "checkpoint_explore", "weights"
    )
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    w_dict = get_weight_dict(chkp, save_folder=weight_folder)
    # print(os.path.abspath(chkp))

    # Prints the nicely formatted dictionary
    # pprint.pprint(w_dict)
