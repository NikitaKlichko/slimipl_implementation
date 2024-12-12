import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
import torch

class DataModule(pl.LightningDataModule):
    def __init__(self, train_datasets, val_datasets, batch_size, n_workers, concat_sampling_probabilities=None):
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.concat_sampling_probabilities = concat_sampling_probabilities
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.collate_fn = self.train_datasets[0]._speech_collate_fn

    def setup(self, stage=None):
        self.concat_train_dataset = ConcatDataset(self.train_datasets)
        self.concat_val_dataset = ConcatDataset(self.val_datasets)

    def train_dataloader(self):
        if self.concat_sampling_probabilities is not None:
            return DataLoader(self.concat_train_dataset , batch_size=self.batch_size, 
                        sampler=WeightedRandomSampler(self.concat_sampling_probabilities, len(self.concat_train_dataset)),
                        collate_fn=self.collate_fn,
                        num_workers=self.n_workers
                        )
        else:
            return DataLoader(self.concat_train_dataset , batch_size=self.batch_size, shuffle=True,
                             collate_fn=self.collate_fn,      
                             num_workers=self.n_workers
                             )

    def val_dataloader(self):
        return DataLoader(self.concat_val_dataset, batch_size=self.batch_size, shuffle=False,
                         collate_fn=self.collate_fn,
                         num_workers=self.n_workers
                         )

class WeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.weights), self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples