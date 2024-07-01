import numpy as np
import h5py
import os


def create_lc_h5py(all_partitions, 
                    all_data, 
                    num_folds, 
                    path_save_dataset):
    
    with h5py.File('{}/dataset.h5'.format(path_save_dataset), 'w') as hf:
        hf.create_dataset("SNID", data = np.array(all_data['oid'].to_numpy()))

        hf.create_dataset("time",  data = np.stack(all_data['time'].to_list(), 0)) 
        hf.create_dataset("flux",     data = np.stack(all_data['flux'].to_list(), 0))
        hf.create_dataset("flux_err",  data = np.stack(all_data['flux_err'].to_list(), 0))
        hf.create_dataset("mask",  data = np.stack(all_data['mask'].to_list(), 0))
        hf.create_dataset("labels",  data = np.stack(all_data['label_int'].to_list(), 0))

        hf.create_dataset("time_detection",  data = np.stack(all_data['time_detection'].to_list(), 0))
        hf.create_dataset("mask_detection",  data = np.stack(all_data['mask_detection'].to_list(), 0))
        hf.create_dataset("time_photometry",  data = np.stack(all_data['time_photometry'].to_list(), 0))
        hf.create_dataset("mask_photometry",  data = np.stack(all_data['mask_photometry'].to_list(), 0))

        for fold in range(num_folds): 
            print("- fold: {}".format(fold))
            aux_pd = all_partitions['fold_%s' % fold]

            hf.create_dataset("training_%d" % fold,
                data = aux_pd[aux_pd['partition'] == 'training_%d' % fold].index.to_numpy())
            hf.create_dataset("validation_%d" % fold,
                data = aux_pd[aux_pd['partition'] == 'validation_%d' % fold].index.to_numpy())
            
        hf.create_dataset("test", 
            data = aux_pd[aux_pd['partition'] == 'test'].index.to_numpy())
        hf.close()

def add_cols_h5py(df, path_save_dataset, name_dataset):
    new_dataset = df.to_numpy()
    with h5py.File('{}/dataset.h5'.format(path_save_dataset), 'a') as hf:
        hf.create_dataset("{}".format(name_dataset), data=new_dataset) #metadata_feat
        hf.close()