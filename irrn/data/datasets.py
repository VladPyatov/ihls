import os.path as osp
from glob import glob
from typing import List

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop


class FromFoldersImageDataset(Dataset):
    """
    Dataset, which loads images in alphabetical order from several different folders.
    """
    def __init__(self, folders_roots: List[str], images_key_names: List[str], grayscale_output: bool = False,
                 images_extensions: List[str] = ['png'], crop_size: int = 256) -> None:
        """
        :param folders_roots: list of folders with desired sets of images
        :param images_key_names: keynames of images from different folders in output dict
        :param grayscale_output: if True, image is converted to return grayscale
        :param images_extensions: which extensions to load
        """
        assert len(folders_roots) == len(images_key_names)
        self.images_paths = {}
        for root, key_name in zip(folders_roots, images_key_names):
            images_in_path = []
            for ext in images_extensions:
                images_in_path += glob(osp.join(root, f'*.{ext}'))
            self.images_paths[key_name] = sorted(images_in_path)

        self.dataset_length = len(self.images_paths[list(self.images_paths.keys())[0]])
        for key in self.images_paths:
            assert len(self.images_paths[key]) == self.dataset_length
        self._grayscale_output = grayscale_output
        self.crop_operator = CenterCrop(crop_size)

    def __getitem__(self, idx):
        out_dict = {}
        for key in self.images_paths:
            image = cv2.imread(self.images_paths[key][idx])
            if image.ndim == 2:
                if self._grayscale_output:
                    image = image[:, :, None]
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if self._grayscale_output:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f'Received unknown image type: {image.__class__}')
            out_dict[key] = self.crop_operator(image)
        return out_dict

    def __len__(self) -> int:
        return self.dataset_length


class DegradedGroundTruthDataset(FromFoldersImageDataset):
    """
    Dataset, which loads degraded images as well as corresponding ground truths in alphabetical order,
    located in different folders.
    """
    def __init__(self, dataset_root_dir: str, crop_size: int = 256, degraded_key_name: str = 'image',
                 ground_truth_key_name: str = 'target', grayscale_output: bool = False,
                 images_extensions: List[str] = ['png'], degraded_folder_name: str = 'degraded',
                 ground_truth_folder_name: str = 'gt'):
        """
        :param dataset_root_dir: root directory with images in subdirectories
        :param degraded_key_name: keyname of degraded images in output dict
        :param ground_truth_key_name: keyname of ground truth images in output dict
        :param degraded_folder_name: name of subfolder with degraded images
        :param ground_truth_folder_name: name of subfolder with ground truth images
        :param grayscale_output: if True, image is converted to return grayscale
        :param images_extensions: which extensions to load
        """
        degraded_path = osp.join(dataset_root_dir, degraded_folder_name)
        ground_truth_path = osp.join(dataset_root_dir, ground_truth_folder_name)
        super(DegradedGroundTruthDataset, self).__init__(
            [degraded_path, ground_truth_path], [degraded_key_name, ground_truth_key_name],
            grayscale_output=grayscale_output, images_extensions=images_extensions, crop_size=crop_size)
