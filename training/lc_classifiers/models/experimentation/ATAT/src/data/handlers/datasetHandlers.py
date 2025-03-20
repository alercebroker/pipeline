from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch


import random

#class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
#    def __init__(self, labels=None):
#        self.labels = labels
#        self.dataset = dict()
#        self.balanced_max = 0
#        # Save all the indices for all the classes
#        for idx in range(0, len(labels)):
#            label = self._get_label(idx)
#            if label not in self.dataset:
#                self.dataset[label] = list()
#            self.dataset[label].append(idx)
#            self.balanced_max = len(self.dataset[label]) \
#                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
#        
#        # Oversample the classes with fewer elements than the max
#        for label in self.dataset:
#            while len(self.dataset[label]) < self.balanced_max:
#                self.dataset[label].append(random.choice(self.dataset[label]))
#        self.keys = list(self.dataset.keys())
#        self.currentkey = 0
#        self.indices = [-1]*len(self.keys)
#
#    def __iter__(self):
#        while self.indices[self.currentkey] < self.balanced_max - 1:
#            self.indices[self.currentkey] += 1
#            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
#            self.currentkey = (self.currentkey + 1) % len(self.keys)
#        self.indices = [-1]*len(self.keys)
#    
#    def _get_label(self, idx, labels = None):
#        return self.labels[idx].item()
#
#    def __len__(self):
#        return self.balanced_max*len(self.keys)
#    

def get_dataloader(
    dataset_used,
    batch_size=128,
    num_workers=8,
    pin_memory=True,
    set_type="train",
    **kwargs
):
    """extra args for Dataloader object"""

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    """ if is data train data must be sampled balanced or not  equally_sample = True (balanced batch)"""

    if set_type == "train":

        # balancedd samples
        class_sample_count = np.array(
            [
                len(np.where(dataset_used.labels == t)[0])
                for t in np.unique(dataset_used.labels)
            ]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in dataset_used.labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"), len(samples_weight)
        )
        #sampler = BalancedBatchSampler(dataset_used.labels)
        loader = DataLoader(dataset_used, sampler=sampler, **loader_kwargs)

    else:
        loader = DataLoader(dataset_used, sampler=None, shuffle=False, **loader_kwargs)

    return loader
