import abc
from typing import Dict, Any

from torch.utils.data import Dataset


class DatasetWrapperBase(Dataset):
    """
    Base class for implementing a dataset wrapper, which adds or changes some data to existing dataset
    """
    dataset: Dataset

    @abc.abstractmethod
    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method returns additional changes to data, which should be added to base dataset output
        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with augmented data
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]
        data = self.augment_data(idx, data)
        return data

    def __len__(self):
        return len(self.dataset)


class CollectionDatasetBase(DatasetWrapperBase):
    """
    This is a base class for collection dataset.
    """
    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return data_dict
