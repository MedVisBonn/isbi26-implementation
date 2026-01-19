import math
import numpy as np
from typing import ( 
    List, 
    Optional,
)
import sys
from time import sleep, time
import threading
from multiprocessing import Event, Process, Queue
import logging
from threadpoolctl import threadpool_limits
from queue import Queue as thrQueue
import random
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, InterpolationMode
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform, 
    MirrorTransform
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, 
    ContrastAugmentationTransform, 
    GammaTransform
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform, 
    RenameTransform, 
    NumpyToTensor
)
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
    LocalSmoothingTransform,
    LocalContrastTransform
)
from batchgenerators.transforms.abstract_transforms import (
    Compose,
    AbstractTransform
)
from batchgenerators.dataloading.multi_threaded_augmenter import producer, results_loop
import lightning as L

from dataset import *




def get_data_module(
    cfg: OmegaConf
):
    if cfg.dataset == 'mnmv2':
        split = getattr(cfg, 'split', None)
        if not split:
            raise ValueError("For dataset=mnmv2, provide a canonical split tag in cfg.split (e.g., 'scanner-symphonytim' or 'phase-exclusive-scanners-all').")
        return MNMv2DataModule(
            data_dir=cfg.data_dir,
            split=split,
            batch_size=cfg.batch_size,
            val_frac=getattr(cfg, 'val_frac', 0.1),
            binary_target=getattr(cfg, 'binary_target', False),
            non_empty_target=getattr(cfg, 'non_empty_target', True),
            train_transforms=cfg.train_transforms,
            seed=getattr(cfg, 'seed', 0)
        )

    elif cfg.dataset == 'pmri':
        # Split-only path: no domain support
        split = getattr(cfg, 'split', None)
        if not split:
            raise ValueError("For dataset=pmri, provide a canonical split tag in cfg.split (e.g., 'promise12-i2cvb').")
        val_size = getattr(cfg, 'val_size', 0.1)
        if split == 'all':
            modules = {}
            for split_name in PMRIDataset.available_splits().keys():
                modules[split_name] = PMRIDataModule(
                    data_dir=cfg.data_dir,
                    split=split_name,
                    batch_size=cfg.batch_size,
                    train_transforms=cfg.train_transforms,
                    val_size=val_size,
                )
            return modules
        return PMRIDataModule(
            data_dir=cfg.data_dir,
            split=split,
            batch_size=cfg.batch_size,
            train_transforms=cfg.train_transforms,
            val_size=val_size,
        )
            

class PMRIDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        data_dir: str,
        split: str,  # name of predefined split (promise12, ...)
        batch_size: int = 32,
        train_transforms: str = 'global_transforms',
        val_size: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_size = val_size
        self.transforms = Transforms(patch_size=[384, 384])
        self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_predict = None, None, None, None

    def prepare_data(self):
        # nothing to prepare for now. Data already downloaded
        pass

    def setup(self, stage=None):
        # Fit / validation setup (split-only)
        if stage in ('fit', None):
            if self.dataset_train is None or self.dataset_val is None:
                train_full = PMRIDataset.from_split(self.data_dir, self.split, 'train', include_meta=True)
                self.dataset_train, self.dataset_val = train_full.random_split(val_size=self.val_size)

        # Test / predict setup
        if stage in ('test', 'predict', None):
            # ensure train/val ready
            if self.dataset_train is None or self.dataset_val is None:
                train_full = PMRIDataset.from_split(self.data_dir, self.split, 'train', include_meta=True)
                self.dataset_train, self.dataset_val = train_full.random_split(val_size=self.val_size)

            if self.dataset_test is None:
                # Build test dataset from predefined split test sites
                self.dataset_test = {f"{self.split}_test": PMRIDataset.from_split(self.data_dir, self.split, 'test', include_meta=True)}
                        


    def train_dataloader(self):
        assert self.dataset_train is not None, "Data not loaded"
        _loader = MultiImageSingleViewDataLoader(
            data=self.dataset_train,
            batch_size=self.batch_size,
            return_orig=False
        )
        
        return MultiThreadedAugmenter(
            data_loader=_loader,
            transform=self.transforms.get_transforms(self.train_transforms),
            num_processes=8,
            num_cached_per_queue = 2, 
            seeds=None
        )
    
    
    def val_dataloader(self):
        assert self.dataset_val is not None, "Data not loaded"
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )

    
    def test_dataloader(self, batch_size: int = None):
        assert self.dataset_train is not None, "Data not loaded"
        assert self.dataset_val is not None, "Data not loaded"
        assert self.dataset_test is not None, "Data not loaded"

        if batch_size is not None:
            self.batch_size = batch_size

        dataloaders = {}

        train_key = f'{self.split}_train'
        val_key   = f'{self.split}_val'

        dataloaders[train_key] = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )

        dataloaders[val_key] = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )
    
        for test_name, dataset in self.dataset_test.items():
            dataloaders[test_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=4
            )

        return dataloaders
    

    def predict_dataloader(self, batch_size: int = None):
        return self.test_dataloader(batch_size=batch_size)
    


class MNMv2DataModule(L.LightningDataModule):
    """Predefined split tag-driven DataModule for MNMv2 (no legacy scenarios).

    Provide a canonical split tag via `split` such as:
      - scanner-symphonytim
      - pathology-dilated-like-vs-hcm-scanners-all
      - phase-exclusive-scanners-symphonytim

    See PREDEFINED_SPLITS for available keys and their semantics.
    """
    # Canonical predefined splits using hyphenated tags
    PREDEFINED_SPLITS = {
        # Scanner source train/val, others test
        'scanner-symphonytim': {'axis': 'scanner', 'source': ['SymphonyTim']},
        'scanner-achieva': {'axis': 'scanner', 'source': ['Achieva']},
        'scanner-signa-hdxt': {'axis': 'scanner', 'source': ['Signa_HDxt']},
        # Pathology protocols (group_A -> train/val, group_B -> test), scanner restriction optional
        'pathology-dilated-like-vs-hcm-scanners-all': {'axis': 'pathology', 'protocol': 'dilated_like_vs_hcm', 'scanners': 'ALL'},
        'pathology-norm-vs-hcm-scanners-all': {'axis': 'pathology', 'protocol': 'norm_vs_hcm', 'scanners': 'ALL'},
        'pathology-norm-vs-arr-scanners-all': {'axis': 'pathology', 'protocol': 'norm_vs_arr', 'scanners': 'ALL'},
        'pathology-norm-vs-fall-scanners-all': {'axis': 'pathology', 'protocol': 'norm_vs_fall', 'scanners': 'ALL'},
        'pathology-fall-vs-hcm-scanners-all': {'axis': 'pathology', 'protocol': 'fall_vs_hcm', 'scanners': 'ALL'},
        'pathology-cia-vs-hcm-scanners-all': {'axis': 'pathology', 'protocol': 'CIA_vs_hcm', 'scanners': 'ALL'},
        'pathology-arr-vs-hcm-scanners-all': {'axis': 'pathology', 'protocol': 'ARR_vs_hcm', 'scanners': 'ALL'},
        # Phase exclusive (ED train/val, ES test) with optional scanner restriction
        'phase-exclusive-scanners-symphonytim': {'axis': 'phase', 'mode': 'exclusive', 'scanners': ['SymphonyTim']},
        'phase-exclusive-scanners-all': {'axis': 'phase', 'mode': 'exclusive', 'scanners': 'ALL'},
    }
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int = 32,
        val_frac: float = 0.1,
        binary_target: bool = False,
        train_transforms: str = 'global_transforms',
        non_empty_target: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.binary_target = binary_target
        self.non_empty_target = non_empty_target
        self.train_transforms = train_transforms
        self.seed = seed
        self.transforms = Transforms(patch_size=[256, 256])
        self.dataset_train = self.dataset_val = self.dataset_test = None
        self._manifest = None
        # Validate split tag
        if self.split not in self.PREDEFINED_SPLITS:
            raise ValueError(f"Unknown MNMv2 split '{self.split}'. Available: {sorted(self.PREDEFINED_SPLITS.keys())}")

    def prepare_data(self):
        pass

    def _build_split(self):
        if self._manifest is None:
            self._manifest = MNMv2Dataset.build_manifest(self.data_dir)
        conf = self.PREDEFINED_SPLITS[self.split]
        axis = conf['axis']
        if axis == 'scanner':
            src = conf.get('source')  # list with single element
            split = MNMv2Dataset.split_domain_shift(
                self._manifest,
                axis='scanner',
                source_values=src,
                val_frac=self.val_frac,
                seed=self.seed
            )
        elif axis == 'pathology':
            proto = conf.get('protocol')
            if not proto:
                raise ValueError('pathology scenario requires protocol=...')
            # Optional scanner restriction (single or ALL)
            scanners = conf.get('scanners')
            manifest = self._manifest
            if scanners and scanners != 'ALL':
                allowed = scanners if isinstance(scanners, list) else scanners
                manifest = manifest[manifest.SCANNER_NORM.isin(allowed)]
                if manifest.empty:
                    raise ValueError('No rows after scanner restriction for pathology scenario')
            split = MNMv2Dataset.split_domain_shift(
                manifest,
                axis='pathology',
                pathology_protocol=proto,
                val_frac=self.val_frac,
                seed=self.seed,
                source_values=None
            )
        elif axis == 'phase':
            mode = conf.get('mode', 'exclusive')
            scanners = conf.get('scanners')
            source_scanners = None
            manifest = self._manifest
            if scanners and scanners != 'ALL':
                source_scanners = scanners if isinstance(scanners, list) else scanners
                manifest = manifest[manifest.SCANNER_NORM.isin(source_scanners)]
                if manifest.empty:
                    raise ValueError('No rows after scanner restriction for phase scenario')
            split = MNMv2Dataset.split_domain_shift(
                manifest,
                axis='phase',
                source_values=source_scanners,
                phase_mode=mode,
                val_frac=self.val_frac,
                seed=self.seed
            )
        else:
            raise ValueError(f"Unsupported scenario axis '{axis}'")
        return split

    def setup(self, stage=None):
        if self.dataset_train is not None and self.dataset_val is not None and self.dataset_test is not None:
            return
        split = self._build_split()
        axis = self.PREDEFINED_SPLITS[self.split]['axis']
        if axis in ('scanner', 'pathology'):
            self.dataset_train = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['train_subjects']),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
            self.dataset_val = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['val_subjects']),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
            self.dataset_test = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['test_subjects']),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
        elif axis == 'phase':
            # Train/val always ED. Test ES.
            self.dataset_train = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['train_subjects']),
                phases=['ED'],
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
            self.dataset_val = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['val_subjects']),
                phases=['ED'],
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
            self.dataset_test = MNMv2Dataset(
                data_dir=self.data_dir,
                manifest=self._manifest,
                subject_codes=list(split['test_subjects']),
                phases=['ES'],
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target
            )
        else:
            raise ValueError('Unhandled axis in setup')

    def train_dataloader(self):
        assert self.dataset_train is not None
        loader = MultiImageSingleViewDataLoader(
            data=self.dataset_train,
            batch_size=self.batch_size,
            return_orig=False
        )
        return MultiThreadedAugmenter(
            data_loader=loader,
            transform=self.transforms.get_transforms(self.train_transforms),
            num_processes=8,
            num_cached_per_queue=2,
            seeds=None
        )

    def val_dataloader(self):
        assert self.dataset_val is not None
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self, batch_size: int = None):
        if batch_size is not None:
            self.batch_size = batch_size
        assert self.dataset_test is not None
        name = self.split
        return {
            f'{name}_train': DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False),
            f'{name}_val': DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False),
            f'{name}_test': DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False),
        }

    def predict_dataloader(self, batch_size: int = None):
        return self.test_dataloader(batch_size=batch_size)


class SingleImageMultiViewDataLoader(SlimDataLoaderBase):
    """Single image multi view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(SingleImageMultiViewDataLoader, self).__init__(data, batch_size)
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # select single slice from dataset
        data = self._data[random.randrange(len(self._data))]
        # split into input and target, cast to np.float for batchgenerators
        img = data['input'].numpy().astype(np.float32)
        tar = data['target'][0].numpy().astype(np.float32)
        # copy along batch dimension
        img_batched = np.tile(img, (self.batch_size, 1, 1, 1))
        tar_batched = np.tile(tar, (self.batch_size, 1, 1, 1))
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img_batched, 
               'seg':  tar_batched}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = data['input'].unsqueeze(0)
            out['target_orig'] = data['target'].unsqueeze(0)
        
        return out
    
    

class MultiImageSingleViewDataLoader(SlimDataLoaderBase):
    """Multi image single view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(MultiImageSingleViewDataLoader, self).__init__(data, batch_size)
        # data is now stored in self._data.
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # get random subset from dataset of batch size length
        # sample = torch.randint(0, len(self._data), size=(self.batch_size,))
        sample = np.random.randint(0, len(self._data), size=(self.batch_size,))
        data   = self._data[sample]
        # split into input and target
        img    = data['input']
        tar    = data['target']
        idx    = data['index']
        
        #construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img.numpy().astype(np.float32), 
               'seg':  tar.numpy().astype(np.float32),
               'index': idx.astype(np.float32)}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = img
            out['target_orig'] = tar
        
        return out
    
    

class Transforms(object):
    """
    A container for organizing and accessing different sets of image transformation operations.

    This class defines four categories of transformations: 'io_transforms', 
    'global_nonspatial_transforms', 'global_transforms', and 'local_transforms'. Each category 
    contains a list of specific transform objects designed for image preprocessing in machine learning tasks.

    The 'io_transforms' are basic input/output operations, like renaming and format conversion.
    'global_nonspatial_transforms' apply transformations like noise addition or resolution simulation 
    that do not depend alter spatial information. 'global_transforms' include spatial transformations 
    like rotation and scaling. 'local_transforms' focus on localized changes to an image, such as adding 
    brightness gradients or local smoothing and are also non-spatial.

    Attributes:
        transforms (dict): A dictionary where keys are transform categories and values are lists of transform objects.

    Methods:
        get_transforms(arg: str): Retrieves a composed transform pipeline based on the specified category.

    Usage:
        >>> transforms = Transforms()
        >>> global_transforms = transforms.get_transforms('global_transforms')
    """
    
    def __init__(
        self,
        patch_size: List[int]
    ) -> None:
        self.patch_size  = patch_size
        io_transforms = [
            RemoveLabelTransform(
                output_key = 'seg', 
                input_key = 'seg',
                replace_with = 0,
                remove_label = -1
            ),
            RenameTransform(
                delete_old = True,
                out_key = 'target',
                in_key = 'seg'
            ),
            RenameTransform(
                delete_old = True,
                out_key = 'input',
                in_key = 'data'
            ),
            NumpyToTensor(
                keys = ['input', 'target'], 
                cast_to = 'float')    
        ]
       
        global_nonspatial_transforms = [
            SimulateLowResolutionTransform(
                order_upsample = 3, 
                order_downsample = 0, 
                channels = None, 
                per_channel = True, 
                p_per_channel = 0.5, 
                p_per_sample = 0.25, 
                data_key = 'data',
                zoom_range = (0.5, 1), 
                ignore_axes = None
            ),
            GaussianNoiseTransform(
                p_per_sample = 0.1, 
                data_key = 'data', 
                noise_variance = (0, 0.1), 
                p_per_channel = 1, 
                per_channel = False
            ),
        ] 
       
        global_transforms = [
            SpatialTransform(
                independent_scale_for_each_axis = False, 
                p_rot_per_sample = 0.2, 
                p_scale_per_sample = 0.2, 
                p_el_per_sample = 0.2, 
                data_key = 'data', 
                label_key = 'seg', 
                patch_size = np.array(self.patch_size), 
                patch_center_dist_from_border = None, 
                do_elastic_deform = False, 
                alpha = (0.0, 200.0), 
                sigma = (9.0, 13.0), 
                do_rotation = True, 
                angle_x = (-3.141592653589793, 3.141592653589793), 
                angle_y = (-0.0, 0.0), 
                angle_z = (-0.0, 0.0), 
                do_scale = True,
                scale = (0.7, 1.4), 
                border_mode_data = 'constant',
                border_cval_data = 0, 
                order_data = 3, 
                border_mode_seg = 'constant',
                border_cval_seg = -1, 
                order_seg = 1,
                random_crop = False,
                p_rot_per_axis = 1, 
                p_independent_scale_per_axis = 1
            ),
            GaussianBlurTransform(
                p_per_sample = 0.2, 
                different_sigma_per_channel = True, 
                p_per_channel = 0.5, 
                data_key = 'data', 
                blur_sigma = (0.5, 1.0), 
                different_sigma_per_axis = False, 
                p_isotropic = 0
            ),
            BrightnessMultiplicativeTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                multiplier_range = (0.75, 1.25), 
                per_channel = True
            ),
            ContrastAugmentationTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                contrast_range = (0.75, 1.25), 
                preserve_range = True, 
                per_channel = True, 
                p_per_channel = 1
            ),
            GammaTransform(
                p_per_sample = 0.1,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = True
            ),
            GammaTransform(
                p_per_sample = 0.3,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = False
            ),
            MirrorTransform(
                p_per_sample = 1, 
                data_key = 'data', 
                label_key = 'seg', 
                axes = (0, 1)
            ),
        ]       
        
        excessive_blurring = [
            SpatialTransform(
                independent_scale_for_each_axis = False, 
                p_rot_per_sample = 0.2, 
                p_scale_per_sample = 0.2, 
                p_el_per_sample = 0.2, 
                data_key = 'data', 
                label_key = 'seg', 
                patch_size = np.array(self.patch_size), 
                patch_center_dist_from_border = None, 
                do_elastic_deform = False, 
                alpha = (0.0, 200.0), 
                sigma = (9.0, 13.0), 
                do_rotation = True, 
                angle_x = (-3.141592653589793, 3.141592653589793), 
                angle_y = (-0.0, 0.0), 
                angle_z = (-0.0, 0.0), 
                do_scale = True,
                scale = (0.7, 1.4), 
                border_mode_data = 'constant',
                border_cval_data = 0, 
                order_data = 3, 
                border_mode_seg = 'constant',
                border_cval_seg = -1, 
                order_seg = 1,
                random_crop = False,
                p_rot_per_axis = 1, 
                p_independent_scale_per_axis = 1
            ),
            GaussianBlurTransform(
                p_per_sample = 0.5, 
                different_sigma_per_channel = True, 
                p_per_channel = 0.5, 
                data_key = 'data', 
                blur_sigma = (3, 5), 
                different_sigma_per_axis = False, 
                p_isotropic = 0
            ),
            BrightnessMultiplicativeTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                multiplier_range = (0.75, 1.25), 
                per_channel = True
            ),
            ContrastAugmentationTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                contrast_range = (0.75, 1.25), 
                preserve_range = True, 
                per_channel = True, 
                p_per_channel = 1
            ),
            GammaTransform(
                p_per_sample = 0.1,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = True
            ),
            GammaTransform(
                p_per_sample = 0.3,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = False
            ),
            MirrorTransform(
                p_per_sample = 1, 
                data_key = 'data', 
                label_key = 'seg', 
                axes = (0, 1)
            ),
        ]

        excessive_noise = [
            GaussianNoiseTransform(
                noise_variance=(0.75,1.25), 
                p_per_sample=0.5, 
                p_per_channel=1, 
                per_channel=False
            ),
            SpatialTransform(
                independent_scale_for_each_axis = False, 
                p_rot_per_sample = 0.2, 
                p_scale_per_sample = 0.2, 
                p_el_per_sample = 0.2, 
                data_key = 'data', 
                label_key = 'seg', 
                patch_size = np.array(self.patch_size), 
                patch_center_dist_from_border = None, 
                do_elastic_deform = False, 
                alpha = (0.0, 200.0), 
                sigma = (9.0, 13.0), 
                do_rotation = True, 
                angle_x = (-3.141592653589793, 3.141592653589793), 
                angle_y = (-0.0, 0.0), 
                angle_z = (-0.0, 0.0), 
                do_scale = True,
                scale = (0.7, 1.4), 
                border_mode_data = 'constant',
                border_cval_data = 0, 
                order_data = 3, 
                border_mode_seg = 'constant',
                border_cval_seg = -1, 
                order_seg = 1,
                random_crop = False,
                p_rot_per_axis = 1, 
                p_independent_scale_per_axis = 1
            ),
            GaussianBlurTransform(
                p_per_sample = 0.2, 
                different_sigma_per_channel = True, 
                p_per_channel = 0.5, 
                data_key = 'data', 
                blur_sigma = (0.5, 1.0), 
                different_sigma_per_axis = False, 
                p_isotropic = 0
            ),
            BrightnessMultiplicativeTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                multiplier_range = (0.75, 1.25), 
                per_channel = True
            ),
            ContrastAugmentationTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                contrast_range = (0.75, 1.25), 
                preserve_range = True, 
                per_channel = True, 
                p_per_channel = 1
            ),
            GammaTransform(
                p_per_sample = 0.1,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = True
            ),
            GammaTransform(
                p_per_sample = 0.3,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = False
            ),
            MirrorTransform(
                p_per_sample = 1, 
                data_key = 'data', 
                label_key = 'seg', 
                axes = (0, 1)
            ),

        ]

        local_transforms = [
            BrightnessGradientAdditiveTransform(
                scale=200, 
                max_strength=4, 
                p_per_sample=0.7, 
                p_per_channel=1
            ),
            LocalGammaTransform(
                scale=200, 
                gamma=(2, 5), 
                p_per_sample=0.7,
                p_per_channel=1
            ),
            LocalSmoothingTransform(
                scale=200,
                smoothing_strength=(0.5, 1),
                p_per_sample=0.7,
                p_per_channel=1
            ),
            LocalContrastTransform(
                scale=200,
                new_contrast=(1, 3),
                p_per_sample=0.7,
                p_per_channel=1
            ),
        ]
        
        self.transforms = {
            'io_transforms': io_transforms,
            'global_nonspatial_transforms': global_nonspatial_transforms + io_transforms,
            'global_transforms': global_transforms + io_transforms,
            'excessive_noise': excessive_noise + io_transforms,
            'excessive_blurring': excessive_blurring + io_transforms,
            'local_transforms': global_nonspatial_transforms + local_transforms + io_transforms,
            'local_val_transforms': local_transforms + io_transforms,
            'all_transforms': local_transforms + global_transforms + io_transforms,
        }


    def get_transforms(
        self, 
        arg: str
    ) -> AbstractTransform:
        if arg in self.transforms:
            return Compose(self.transforms[arg])

        elif 'global_without' in arg:
            print(arg.split('_')[-1])
            missing_transform = arg.split('_')[-1]
            return Compose(list(
                filter(
                    lambda x: x.__class__.__name__ != missing_transform,
                    self.transforms['global_transforms']
                )
            ))
        
        else:
            raise ValueError(f"Unknown transform category {arg}. Must be one of {self.transforms.keys()}")



def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Zoom":
        scale = (100 - magnitude) / 100
        img = F.resized_crop(
            img,
            top=int(img.shape[-2] // 2 - (img.shape[-2] * scale) // 2),
            left=int(img.shape[-1] // 2 - (img.shape[-1] * scale) // 2),
            height=int(img.shape[-2] * scale),
            width=int(img.shape[-1] * scale),
            size=img.shape[-2:],
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



def volume_collate(batch: List[dict]) -> dict:
    return batch[0]



class MultiThreadedAugmenter(object):
    """ 
    Adapted from batchgenerators, see https://github.com/MIC-DKFZ/batchgenerators/
    Changed user_api from blas to openmp in class method _start in threadpool_limits.
    Otherwise it doesn't work with docker!
    
    Makes your pipeline multi threaded. Yeah!
    If seeded we guarantee that batches are retunred in the same order and with the same augmentation every time this
    is run. This is realized internally by using une queue per worker and querying the queues one ofter the other.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure
        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
        num_processes (int): number of processes
        num_cached_per_queue (int): number of batches cached per process (each process has its own
        multiprocessing.Queue). We found 2 to be ideal.
        seeds (list of int): one seed for each worker. Must have len(num_processes).
        If None then seeds = range(num_processes)
        pin_memory (bool): set to True if all torch tensors in data_dict are to be pinned. Pytorch only.
        timeout (int): How long do we wait for the background workers to do stuff? If timeout seconds have passed and
        self.__get_next_item still has not gotten an item from the workers we will perform a check whether all
        background workers are still alive. If all are alive we wait, if not we set the abort flag.
        wait_time (float): set this to be lower than the time you need per iteration. Don't set this to 0,
        that will come with a performance penalty. Default is 0.02 which will be fine for 50 iterations/s
    """

    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False,
                 timeout=10, wait_time=0.02):
        self.timeout = timeout
        self.pin_memory = pin_memory
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._processes = []
        self._end_ctr = 0
        self._queue_ctr = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None
        self.abort_event = Event()
        self.wait_time = wait_time
        self.was_initialized = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __get_next_item(self):
        item = None

        while item is None:
            if self.abort_event.is_set():
                self._finish()
                raise RuntimeError("One or more background workers are no longer alive. Exiting. Please check the "
                                   "print statements above for the actual error message")

            if not self.pin_memory_queue.empty():
                item = self.pin_memory_queue.get()
            else:
                sleep(self.wait_time)

        return item

    def __next__(self):
        if not self.was_initialized:
            self._start()

        try:
            item = self.__get_next_item()

            while isinstance(item, str) and (item == "end"):
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_ctr = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    raise StopIteration

                item = self.__get_next_item()

            return item

        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self.abort_event.set()
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if not self.was_initialized:
            self._finish()
            self.abort_event.clear()

            logging.debug("starting workers")
            self._queue_ctr = 0
            self._end_ctr = 0

            if hasattr(self.generator, 'was_initialized'):
                self.generator.was_initialized = False

            with threadpool_limits(limits=1, user_api="openmp"):
                for i in range(self.num_processes):
                    self._queues.append(Queue(self.num_cached_per_queue))
                    self._processes.append(Process(target=producer, args=(
                        self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event)))
                    self._processes[-1].daemon = True
                    self._processes[-1].start()

            if torch is not None and torch.cuda.is_available():
                gpu = torch.cuda.current_device()
            else:
                gpu = None

            # more caching = more performance. But don't cache too much or your RAM will hate you
            self.pin_memory_queue = thrQueue(max(3, self.num_cached_per_queue * self.num_processes // 2))

            self.pin_memory_thread = threading.Thread(target=results_loop, args=(
                self._queues, self.pin_memory_queue, self.abort_event, self.pin_memory, gpu, self.wait_time,
                self._processes))

            self.pin_memory_thread.daemon = True
            self.pin_memory_thread.start()

            self.was_initialized = True
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but it has already been "
                          "initialized previously")

    def _finish(self, timeout=10):
        self.abort_event.set()

        start = time()
        while self.pin_memory_thread is not None and self.pin_memory_thread.is_alive() and start + timeout > time():
            
            sleep(0.2)

        if len(self._processes) != 0:
            logging.debug("MultiThreadedGenerator: shutting down workers...")
            [i.terminate() for i in self._processes]

            for i, p in enumerate(self._processes):
                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_ctr = 0

            del self.pin_memory_queue
        self.was_initialized = False

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
    
