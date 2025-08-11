from typing import List

from .datasets import DegradedGroundTruthDataset
from .wrappers import KernelDatasetWrapper, NoiseDatasetWrapper, ImagesPreparationWrapper
import os.path as osp
from .base import CollectionDatasetBase


class BlurDataset(CollectionDatasetBase):
    """
    Dataset, which loads noisy degraded images as well as corresponding ground truths and kernels in alphabetical order,
    located in different folders.
    """
    def __init__(self, dataset_root_dir: str, min_noise_std: float, max_noise_std: float, crop_size: int = 256,
                 degraded_key_name: str = 'image', ground_truth_key_name: str = 'target',
                 kernel_key_name: str = 'kernel', noise_std_key_name: str = 'noise_std',
                 grayscale_output: bool = False, images_extensions: List[str] = ['png'],
                 degraded_folder_name: str = 'degraded', ground_truth_folder_name: str = 'gt',
                 kernel_folder_name: str = 'kernels'):
        dataset = DegradedGroundTruthDataset(dataset_root_dir, crop_size=crop_size, degraded_key_name=degraded_key_name,
                                             ground_truth_key_name=ground_truth_key_name,
                                             grayscale_output=grayscale_output,
                                             images_extensions=images_extensions,
                                             degraded_folder_name=degraded_folder_name,
                                             ground_truth_folder_name=ground_truth_folder_name)
        dataset = KernelDatasetWrapper(dataset, osp.join(dataset_root_dir, kernel_folder_name),
                                       kernel_key_name=kernel_key_name)
        dataset = NoiseDatasetWrapper(dataset, min_noise_std, max_noise_std, degraded_key_name,
                                      noise_std_key_name=noise_std_key_name)
        self.dataset = ImagesPreparationWrapper(dataset,
                                                key_names_to_process=[degraded_key_name, ground_truth_key_name])
