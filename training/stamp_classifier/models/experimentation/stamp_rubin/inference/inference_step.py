import tensorflow as tf
import pandas as pd

from src.models.CNN_model import DynamicStampModel
from src.data.stamp_processing import prepare_model_input, get_max_hw,normalize_stamps
from src.data.metadata_processing import process_coordinates, apply_normalization

def process_stamp(data,args):
    stamp_cols = args['stamps_cols']
    stamps = data[stamp_cols].values
    max_h, max_w = get_max_hw(stamps)
    padded_stamps, padding_masks = prepare_model_input(stamps, max_h, max_w)
    padded_stamps = normalize_stamps(padded_stamps, padding_masks)
    return padded_stamps

def process_metadata(
        data, 
        args, 
        dict_info_model, 
        norm_type='z-score', 
        path_norm_dir='./normalization_params', 
        is_test_only=False
        ):
    
    
    columns_to_rm = args['stamps_cols'] + [args['candid_col']] + ["target_name"] # nuevo
    metadata = data.drop(columns=columns_to_rm, axis=1)
    metadata = metadata.set_index(args['id_col'])

    coord_df = process_coordinates(
        oids=metadata.index,
        ra=metadata[args['ra_col']].values,
        dec=metadata[args['dec_col']].values,
        coord_type=args['coord_type'],
    )
    #por ahora estamos botando las coordenadas
    #metadata = pd.concat([metadata, coord_df], axis=1)
    metadata = metadata.drop(columns=[args['ra_col'], args['dec_col']])
    metadata = metadata.fillna(-999)

    ordered_cols = dict_info_model['order_features']
    metadata = metadata[ordered_cols]

    metadata = apply_normalization(
        metadata,
        norm_type=norm_type,
        dict_info_model=dict_info_model,
        path_norm_dir=path_norm_dir,
        is_test_only=is_test_only
    )
    return metadata


def perform_inference(data_path, model_path,zscore_path,args,dict_info_model):

    #tengo que leer la data
    #tengo que leer el zscore_path
    #tengo que transformar la data
    #tengo que cargar el modelo
    #dejar la configuracion harcodeada
    #tengo que hacer la prediccion
    #tengo que guardar el resultado

    data = pd.read_pickle(data_path)
    #print(data.columns)
    stamps = process_stamp(data,args)


    md_test = process_metadata(
        data, args, dict_info_model,
        norm_type=args['norm_type'], path_norm_dir=zscore_path, 
        is_test_only=True
    )
    #print(md_test.head())

    dataset = tf.data.Dataset.from_tensors((stamps, md_test))
    dataset = dataset.unbatch().batch(32).prefetch(5)

    stamp_classifier = tf.keras.models.load_model(
            model_path, custom_objects={"DynamicStampModel": DynamicStampModel}
        )

    predictions = []
    for (samples, md) in dataset:
        logits = stamp_classifier((samples, md), training=False)
        predictions.append(logits)
    predictions = tf.concat(predictions, axis=0)
    print(predictions.shape)
    return predictions

if __name__ == "__main__":
    args = {'stamps_cols': ['visit_image','difference_image','reference_image'], 'candid_col': 'diaSourceId',
             'id_col': 'diaObjectId', 'ra_col': 'ra', 
             'dec_col': 'dec', 'coord_type': 'spherical', 
             'norm_type': 'z-score'}
    
    dict_info_model = {'order_features': ["airmass","magLim","psfFlux","psfFluxErr",
                                          'scienceFlux','scienceFluxErr','seeing','snr']}

    perform_inference(data_path = './chunk0000_stamps.pkl',model_path = './model.keras',zscore_path= './',
                      args=args, dict_info_model=dict_info_model) 