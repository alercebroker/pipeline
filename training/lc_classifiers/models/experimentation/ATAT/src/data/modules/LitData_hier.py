import lightning as L
import os

from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.CustomDataset_hier import ATATDataset
from utils import save_yaml, load_yaml

class LitData(L.LightningDataModule):
    def __init__(self, path_results: str, fold: int, data_root: str, batch_size: int, **kwargs):
        super().__init__()
        self.path_results = path_results
        self.fold = fold
        self.data_root = data_root
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        dict_info_path = os.path.join(data_root, 'dict_info.yaml')
        self.dict_info = load_yaml(dict_info_path)
        self.list_time_to_eval = self.dict_info['list_time_to_eval']
        self.feat_cols = self.dict_info.get('feat_cols', None)
        self.md_cols = self.dict_info.get('md_cols', None)
        if not os.path.exists(dict_info_path):
            save_yaml(self.dict_info, dict_info_path)
        
    def setup(self, stage: str = None):
        """
        Setup datasets for different stages: 'fit', 'validate', 'test'.
        """
        if stage in ["fit", "train"] or stage is None:
            self.train_dataset = ATATDataset(
                path_results=self.path_results, data_root=self.data_root, 
                set_type="train", fold=self.fold, feat_cols=self.feat_cols, 
                **self.kwargs
            )
        
        if stage in ["fit", "validation", "val"] or stage is None:
            self.val_dataset = ATATDataset(
                path_results=self.path_results, data_root=self.data_root, 
                set_type="validation", fold=self.fold, feat_cols=self.feat_cols, 
                **self.kwargs
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = ATATDataset(
                path_results=self.path_results, data_root=self.data_root, 
                set_type="test", fold=self.fold, feat_cols=self.feat_cols, 
                **self.kwargs
            )

    def train_dataloader(self):
        return get_dataloader(dataset_used=self.train_dataset, set_type="train")

    def val_dataloader(self):
        return get_dataloader(dataset_used=self.val_dataset, set_type="validation")

    def test_dataloader(self):
        """
        Return multiple DataLoaders, one for each time_to_eval in list_time_to_eval.
        """
        dataloaders = []
        for time_to_eval in self.list_time_to_eval:
            dataset = self._filter_test_dataset_by_time(self.test_dataset, time_to_eval)
            dataloaders.append(get_dataloader(dataset, set_type="test", batch_size=self.batch_size))
        return dataloaders

    def predict_dataloader(self, stage='test'):
        if stage == 'train':
            return get_dataloader(dataset_used=self.train_dataset, set_type="train")
        elif stage == 'val':
            return get_dataloader(dataset_used=self.val_dataset, set_type="validation")
        else: 
            return get_dataloader(dataset_used=self.test_dataset, set_type="test")

    def _filter_test_dataset_by_time(self, dataset, time_to_eval):
        """
        Apply `obtain_valid_mask` for a specific time_to_eval to create a filtered dataset.
        """
        filtered_data = []
        for sample in dataset:
            filtered_sample = dataset.obtain_valid_mask(sample, time_to_eval, sample['idx'])
            filtered_data.append(filtered_sample)
        return filtered_data

