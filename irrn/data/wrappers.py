import os.path as osp
from glob import glob
from typing import Dict, Any, List

import numpy as np
from torch.utils.data import Dataset

from .base import DatasetWrapperBase


class ImagesPreparationWrapper(DatasetWrapperBase):
    """
    Wrapper, which processes images in dataset: converts to float representation and permutes dimensions.
    """
    def __init__(self, dataset: Dataset, key_names_to_process: List[str] = ['image', 'target']):
        """
        :param dataset: dataset with images to be transformed
        :param key_names_to_process: which key names to process in parent dataset
        """
        self.dataset = dataset
        self._images_key_names = key_names_to_process

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare images for forward pass

        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with transformed images
        """
        for key in self._images_key_names:
            data_dict[key] = data_dict[key].transpose(2, 0, 1).astype('float32')/255
        return data_dict


class NoiseDatasetWrapper(DatasetWrapperBase):
    """
    Wrapper, which adds noise stds to dataset
    """
    def __init__(self, dataset: Dataset, min_std: float, max_std: float, image_key_name: str,
                 noise_std_key_name: str = 'noise_std') -> None:
        """
        :param dataset: dataset to wrap around the noise sampling
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        :param image_key_name: key name of image in dict, that should be degraded with noise
        :param noise_std_key_name: key name of noise std, which is added to the output dict
        """
        self.min_std = min_std
        self.max_std = max_std
        self.dataset = dataset
        self._image_key_name = image_key_name
        self._noise_key_name = noise_std_key_name

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add noise std value to data dict
        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with added noise std value
        """
        image = data_dict[self._image_key_name].astype('float')

        noise_std = self.min_std + np.random.rand()*(self.max_std - self.min_std)

        image += noise_std*np.random.standard_normal(size=image.shape)
        image = np.clip(image, 0, 255).astype('uint8')

        data_dict.update({self._image_key_name: image, self._noise_key_name: noise_std})
        return data_dict


class KernelDatasetWrapper(DatasetWrapperBase):
    """
    Wrapper, which adds kernels to dataset
    """
    def __init__(self, dataset: Dataset, path_to_kernels: str, kernel_key_name: str = 'kernel') -> None:
        """
        :param dataset: dataset to wrap around the kernels loading
        :param path_to_kernels: path of folder with sampled kernels
        """
        self.kernels_paths = sorted(glob(osp.join(path_to_kernels, '*.npy')))
        assert len(dataset) == len(self.kernels_paths)
        self._kernel_key_name = kernel_key_name
        self.dataset = dataset

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add downscaling kernel to dataset
        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with added downscale kernels
        """
        kernel = self.kernels_paths[idx]
        kernel = np.load(kernel)
        data_dict.update({self._kernel_key_name: kernel})
        return data_dict
