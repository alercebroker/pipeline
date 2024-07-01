import pytorch_lightning as pl
from elasticc import get_dataloader, ElasticcAug


class LitDataAugmented(pl.LightningDataModule):
    def __init__(self, data_root: str = "path/to/dir", batch_size: int = 32, **kwargs):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.kwargs = kwargs

    def train_dataloader(self):
        return get_dataloader(
            dataset_used=ElasticcAug(
                data_root=self.data_root, set_type="train", **self.kwargs
            ),
            set_type="train",
        )

    def val_dataloader(self):
        return get_dataloader(
            dataset_used=ElasticcAug(
                data_root=self.data_root, set_type="validation", **self.kwargs
            ),
            set_type="validation",
        )
