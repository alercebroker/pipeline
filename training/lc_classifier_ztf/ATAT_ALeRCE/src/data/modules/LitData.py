import pytorch_lightning as pl

from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.CustomDataset import ATATDataset


class LitData(pl.LightningDataModule):
    def __init__(self, data_root: str = "path/to/dir", batch_size: int = 32, **kwargs):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.kwargs = kwargs

    def train_dataloader(self):
        return get_dataloader(
            dataset_used=ATATDataset(
                data_root=self.data_root, set_type="train", **self.kwargs
            ),
            set_type="train",
        )

    def val_dataloader(self):
        return get_dataloader(
            dataset_used=ATATDataset(
                data_root=self.data_root, set_type="validation", **self.kwargs
            ),
            set_type="validation",
        )

    def test_dataloader(self):
        return get_dataloader(
            dataset_used=ATATDataset(
                data_root=self.data_root, set_type="test", **self.kwargs
            ),
            set_type="test",
        )
