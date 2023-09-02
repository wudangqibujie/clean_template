import pandas as pd
from torch.utils.data import IterableDataset, DataLoader, Dataset, get_worker_info
from pathlib import Path
import os


class InMemPandasDataSet(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, item):
        return self.df.iloc[item: item + 1, :]

    def __len__(self):
        return self.df.shape[0]


class IterPandasDataSet(IterableDataset):
    def __init__(self, data_folder, prefix=''):
        self.files_path = self._files_path(data_folder, prefix)

    def _files_path(self, data_folder, prefix=''):
        return [Path(data_folder) / f for f in os.listdir(data_folder) if prefix and f.startswith(prefix)]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            block_tasks = self.files_path
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            block_tasks = [f for f in self.files_path if hash(f) % num_workers == worker_id]
        return self._wrapper_iter(block_tasks)

    def _wrapper_iter(self, paths):
        for path in paths:
            df = pd.read_csv(path, chunksize=1)
            yield df


class HiveDataSet(IterableDataset):
    def __init__(self):
        pass