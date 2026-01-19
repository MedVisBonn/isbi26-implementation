"""
This script collects the Dataset classes for the PMRI and MNMv2 datasets.

Usage: serves only as a collection of individual functionalities
Authors: Jonathan Lennartz
"""

# - standard packages
from pathlib import Path
import random
from functools import lru_cache
# - third party packages
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,

)
import nibabel as nib
# - local source


class PMRIDataset(Dataset):
    """
    Multi-site dataset for prostate MRI segmentation from:
    https://liuquande.github.io/SAML/
    Voxel Spacing: 
        Domain: RUNMC, Min Spacing 0.500, Max Spacing 0.521, 419
        Domain: BMC, Min Spacing 0.417, Max Spacing 0.443, 324
        Domain: I2CVB, Min Spacing 0.534, Max Spacing 0.617, 505
        Domain: UCL, Min Spacing 0.469, Max Spacing 0.521, 171
        Domain: BIDMC, Min Spacing 0.365, Max Spacing 0.417, 197
        Domain: HK, Min Spacing 0.521, Max Spacing 0.521, 157

    Possible domains:
        ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
    Initialization parameters:
    - data_dir -> Path to dataset directory
    - vendor -> Vendor from possible ones to load data
    """

    _DS_CONFIG = {"num_classes": 2, "spatial_dims": 2, "size": (384, 384)}

    # Mapping of site letters (A-F) to site metadata & folder names
    SITES = {
        # letter:   (folder_name , metadata dict)
        "A": ("RUNMC", {"dataset": "NCI-ISBI 2013", "institution": "RUNMC",  "cases": 30, "field_strength_T": 3.0,        "coil": "surface",     "vendor": "Siemens"}),
        "B": ("BMC",   {"dataset": "NCI-ISBI 2013", "institution": "BMC",    "cases": 30, "field_strength_T": 1.5,        "coil": "endorectal",  "vendor": "Philips"}),
        "C": ("I2CVB", {"dataset": "I2CVB",         "institution": "HCRUDB", "cases": 19, "field_strength_T": 3.0,        "coil": "none",        "vendor": "Siemens"}),
        "D": ("UCL",   {"dataset": "PROMISE12",     "institution": "UCL",    "cases": 13, "field_strength_T": [1.5, 3.0], "coil": "none",        "vendor": "Siemens"}),
        "E": ("BIDMC", {"dataset": "PROMISE12",     "institution": "BIDMC",  "cases": 12, "field_strength_T": 3.0,        "coil": "endorectal",  "vendor": "GE"}),
        "F": ("HK",    {"dataset": "PROMISE12",     "institution": "HK",     "cases": 12, "field_strength_T": 1.5,        "coil": "endorectal",  "vendor": "Siemens"}),
    }

    PREDEFINED_SPLITS = {
        # Canonical hyphenated names only (lowercase, '-' inside parts)
        "promise12-i2cvb": {
            "train": ["C", "D", "E", "F"],
            "test":  ["A", "B"],
            "notes": "Train PROMISE12 + I2CVB; hold out NCI-ISBI 2013."
        },
        "promise12-nci-isbi": {
            "train": ["A", "B", "D", "E", "F"],
            "test":  ["C"],
            "notes": "Train PROMISE12 + NCI-ISBI; hold out I2CVB."
        },
        "promise12": {
            "train": ["D", "E", "F"],
            "test":  ["A", "B", "C"],
            "notes": "Train PROMISE12; hold out ISBI+I2CVB."
        },
        "threet-to-onepointfivet": {
            "train": ["A", "C", "E"],
            "test":  ["B", "F"],
            "notes": "Omit D (mixed 1.5/3T) to keep B0 pure."
        },
        "onepointfivet-to-threet": {
            "train": ["B", "F"],
            "test":  ["A", "C", "E"],
            "notes": "Omit D (mixed 1.5/3T) to keep B0 pure."
        },
        "noerc-to-erc": {
            "train": ["A", "C", "D"],
            "test":  ["B", "E", "F"],
            "notes": "No endorectal coil -> endorectal coil."
        }
    }

    def __init__(
        self,
        data_dir: str,
        domain: str = None,  # backwards compatibility: single site string e.g. "RUNMC"
        *,
        sites=None,          # Optional list/tuple of site letters (A-F) OR folder names
        split: str = None,   # Name of predefined split in PREDEFINED_SPLITS
        split_part: str = None,  # 'train' or 'test' when using split
        non_empty_target: bool = True,
        normalize: bool = True,
        include_meta: bool = False,  # deprecated: retained for backward-compat, no per-sample return
    ):
        """Initialize a (multi-)site PMRI slice dataset.

        Parameters
        ----------
        data_dir : str
            Root directory containing site sub-folders (RUNMC, BMC, ...)
        domain : str, optional
            Legacy parameter for single-site loading (ignored if `sites` or `split` provided).
        sites : list[str] | str, optional
            Explicit list of sites to include. Accepts site letters (A-F) or folder names.
        split : str, optional
            Name of a predefined split from `PREDEFINED_SPLITS`.
        split_part : str, optional
            Which partition of the split to load ('train' or 'test'). Required if `split` is used.
        non_empty_target : bool
            Remove slices without any target label.
        normalize : bool
            Normalize per-dataset using global mean/std.
        include_meta : bool
            Store per-slice metadata (site_letter, site_name, vendor, coil, field_strength_T, split, split_part).
        """

        if split is not None and (sites is not None or domain is not None):
            raise ValueError("Provide either `split` or (`sites`/`domain`), not both.")
        if split is not None:
            if split not in self.PREDEFINED_SPLITS:
                raise ValueError(f"Unknown split '{split}'. Available: {list(self.PREDEFINED_SPLITS)}")
            if split_part not in ("train", "test"):
                raise ValueError("`split_part` must be 'train' or 'test' when using a split.")
            sites = self.PREDEFINED_SPLITS[split][split_part]
        elif sites is None and domain is not None:
            sites = [domain]
        elif sites is None and domain is None:
            raise ValueError("You must provide one of: domain, sites, or split.")

        # Normalize sites to folder names and keep letter mapping
        if isinstance(sites, (str, Path)):
            sites = [str(sites)]
        normalized_sites = []  # list of (folder_name, letter or None)
        for s in sites:
            s_upper = str(s).upper()
            if s_upper in self.SITES:  # letter
                folder_name = self.SITES[s_upper][0]
                normalized_sites.append((folder_name, s_upper))
            else:
                # try to match folder name in SITES mapping values
                matched_letter = None
                for letter, (folder, _meta) in self.SITES.items():
                    if folder.upper() == s_upper:
                        matched_letter = letter
                        folder_name = folder
                        break
                if matched_letter is None:
                    raise ValueError(f"Unknown site identifier '{s}'. Use letters A-F or one of {[v[0] for v in self.SITES.values()]}")
                normalized_sites.append((folder_name, matched_letter))

        self.sites = normalized_sites  # list of tuples (folder_name, letter)
        self.split = split
        self.split_part = split_part
        self._data_dir = Path(data_dir).resolve()
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._include_meta = include_meta  # kept for backward compatibility (no longer returned per sample)
        self.target_spacing = 0.5
        self._crop = CenterCrop(384)
        # backward compatibility attributes
        self.domains = [folder for folder, _ in self.sites]
        # if a single site, keep a scalar attribute as before
        self.domain = self.domains[0] if len(self.domains) == 1 else self.domains
        self._load_data()


    def _load_data(self):
        self.input = []
        self.target = []
        self.slice_site_letters = []  # parallel list to slices after concatenation (will filter with masks)
        slice_meta = []  # legacy (only if include_meta True)
        # Track per-case slice counts to support subject-safe splits
        case_slice_counts = []
        case_ids = []

        for folder_name, letter in self.sites:
            site_path = self._data_dir / folder_name
            if not site_path.exists():
                raise FileNotFoundError(f"Expected site folder '{folder_name}' at {site_path}")
            # iterate files for this site
            for file in site_path.iterdir():
                if "segmentation" in file.name.lower():
                    case = file.name[4:6]
                    seg_name = "Segmentation" if folder_name == "BMC" else "segmentation"
                    case_input_path = site_path / f"Case{case}.nii.gz"
                    case_target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
                    x_img = nib.load(case_input_path)
                    y_img = nib.load(case_target_path)

                    spacing = x_img.header.get_zooms()[0]
                    x = torch.tensor(x_img.get_fdata())  # (H,W,D)
                    y = torch.tensor(y_img.get_fdata(), dtype=torch.long)

                    scale_factor = (1 / self.target_spacing * spacing, 1 / self.target_spacing * spacing)
                    x = interpolate(
                        x.unsqueeze(1),
                        scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)
                    y = interpolate(
                        y.unsqueeze(1).float(),
                        scale_factor=scale_factor,
                        mode='nearest'
                    ).long().squeeze(1)

                    x = self._crop(x)
                    y = self._crop(y)

                    # Append data and track case slice count/id for subject-safe splitting
                    self.input.append(x)
                    self.target.append(y)
                    case_slice_counts.append(x.shape[-1])
                    case_ids.append(f"{folder_name}_Case{case}")
                    # track site letters per slice
                    self.slice_site_letters.extend([letter] * x.shape[-1])
                    if self._include_meta:
                        # legacy meta collection (not returned per sample)
                        n_slices = x.shape[-1]
                        meta_base = {}
                        if letter in self.SITES:
                            meta_base.update(self.SITES[letter][1])
                        meta_base.update({
                            "site_letter": letter,
                            "site_name": folder_name,
                            "case_id": f"{folder_name}_Case{case}",
                            "split": self.split,
                            "split_part": self.split_part,
                        })
                        slice_meta.extend([meta_base] * n_slices)

        # Concatenate / reshape
        if len(self.input) == 0:
            raise RuntimeError("No data loaded. Check site selections and data directory.")
        self.input = torch.cat(self.input, dim=-1).moveaxis(-1, 0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=-1).moveaxis(-1, 0).unsqueeze(1)
        self.target[self.target == 2] = 1  # merge prostate label variants

        # Build subject-level slice index buckets (after optional non-empty filtering)
        # First compute original per-case index ranges
        starts = [0] + np.cumsum(case_slice_counts[:-1]).tolist() if len(case_slice_counts) > 0 else []
        ends = np.cumsum(case_slice_counts).tolist() if len(case_slice_counts) > 0 else []
        # Optionally filter out empty-target slices while preserving per-case grouping
        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            # Apply mask to tensors and parallel lists
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]
            self.slice_site_letters = [s for i, s in enumerate(self.slice_site_letters) if non_empty_slices[i].item()]
            if self._include_meta:
                slice_meta = [m for i, m in enumerate(slice_meta) if non_empty_slices[i].item()]
            # Recompute per-case indices after filtering by building contiguous ranges in filtered tensor
            subjects_idx = []
            filtered_case_ids = []
            offset = 0
            for start, end, cid in zip(starts, ends, case_ids):
                if end <= start:
                    continue
                local_mask = non_empty_slices[start:end]
                kept = int(local_mask.sum().item())
                if kept > 0:
                    subjects_idx.append(torch.arange(offset, offset + kept, dtype=torch.long))
                    filtered_case_ids.append(cid)
                    offset += kept
            self.subjects_slice_indices = subjects_idx
            self.subject_ids = filtered_case_ids
        else:
            # No filtering: build contiguous ranges
            self.subjects_slice_indices = [torch.arange(s, e) for s, e in zip(starts, ends)]
            self.subject_ids = list(case_ids)

        if self._normalize:
            mean = self.input.mean()
            std = self.input.std()
            self.input = (self.input - mean) / std

        self.meta = slice_meta if self._include_meta else None

    def site_slice_distribution(self):
        """Return a dict counting slices per site letter."""
        from collections import Counter
        return dict(Counter(self.slice_site_letters))

    def site_letters(self):
        """Return the set of site letters present in this dataset."""
        return set(self.slice_site_letters)

    @classmethod
    def available_splits(cls):
        """Return a dictionary of available predefined splits (names -> definition)."""
        return cls.PREDEFINED_SPLITS

    @classmethod
    def from_split(cls, data_dir: str, split: str, part: str, **kwargs):
        """Convenience constructor for predefined splits.

        Example
        -------
    train_ds = PMRIDataset.from_split(root, 'promise12', 'train')
    test_ds  = PMRIDataset.from_split(root, 'promise12', 'test')
        """
        return cls(data_dir=data_dir, split=split, split_part=part, **kwargs)


    def random_split(
        self,
        val_size: float = 0.2,
    ):
        class PMRISubset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]

            def __getitem__(self, idx):
                sample = {
                    "input": self.input[idx],
                    "target": self.target[idx],
                    "index": idx,
                }
                if hasattr(self, 'meta') and self.meta is not None:
                    sample.update(self.meta[idx])
                return sample
            
        # Subject-safe, reproducible split using per-case index buckets
        torch.manual_seed(0)
        n_subjects = len(self.subjects_slice_indices)
        if n_subjects == 0:
            raise RuntimeError("No subjects available for splitting in PMRI dataset.")
        perm = torch.randperm(n_subjects).tolist()
        n_val = int(round(val_size * n_subjects))
        val_subjects = set(perm[:n_val])
        train_subjects = set(perm[n_val:])

        def flatten_indices(idxs):
            if not idxs:
                return torch.tensor([], dtype=torch.long)
            return torch.cat([self.subjects_slice_indices[i] for i in idxs], dim=0)

        train_idx = flatten_indices(sorted(list(train_subjects)))
        val_idx = flatten_indices(sorted(list(val_subjects)))

        dataset_train = PMRISubset(
            input=self.input[train_idx],
            target=self.target[train_idx],
        )

        dataset_val = PMRISubset(
            input=self.input[val_idx],
            target=self.target[val_idx],
        )

        return dataset_train, dataset_val


    def __len__(self):
        return self.input.shape[0]
    

    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
            "index": idx
        }
    


class MNMv2Dataset(Dataset):
    """
    Vendors: 
        Siemens:
            Domain: Symphony, Min Spacing 1.000, Max Spacing 1.641
                Number of cases: 4024
            Domain: Trio, Min Spacing 1.000, Max Spacing 1.328
                Number of cases: 128
            Domain: Avanto, Min Spacing 0.977, Max Spacing 1.417
                Number of cases: 904 
                
        GE (Signa): 
            Domain: HDxt, Min Spacing 0.605, Max Spacing 1.563
                Number of cases: 618
            Domain: EXCITE, Min Spacing 1.000, Max Spacing 1.484
                Number of cases: 632
            Domain: Explorer, Min Spacing 0.781, Max Spacing 0.781
                Number of cases: 26

        Philips:
            Domain: Achieva, Min Spacing 0.684, Max Spacing 1.484
                Number of cases: 1796 

        As List: ["Symphony", "Trio", "Avanto", "HDxt", "EXCITE", "Explorer", "Achieva"]    
    """
    # siemens Min Spacing 0.977, Max Spacing 1.641 SymphonyTim, TrioTim, Avanto Fit,
    # GE Min Spacing 0.605, Max Spacing 1.563 Signa HDxt, SIGNA EXCITE, Signa Explorer
    # philips Min Spacing 0.684, Max Spacing 1.484  Achieva

    # ---- New manifest & split utilities ----
    _SCANNER_RENAME = {
        'Achieva': 'Achieva',
        'SIGNA EXCITE': 'Signa_Excite',
        'SIGNA HDxt': 'Signa_HDxt',
        'Signa HDxt': 'Signa_HDxt',
        'SIGNA Explorer': 'Signa_Explorer',
        'Signa Explorer': 'Signa_Explorer',
        'Avanto': 'Avanto',
        'Avanto Fit': 'Avanto_Fit',
        'Symphony': 'Symphony',
        'SymphonyTim': 'SymphonyTim',
        'TrioTim': 'TrioTim',
    }

    EXCLUDE_PATHOLOGIES = {'DRV', 'TRI'}

    PATHOLOGY_PROTOCOLS = {
        'norm_vs_fall': {
            'group_A': ['NOR'],
            'group_B': ['FALL']
        },
        'norm_vs_hcm': {
            'group_A': ['NOR'],
            'group_B': ['HCM'],
        },
        'norm_vs_arr': {
            'group_A': ['NOR'],
            'group_B': ['ARR'],
        },
        'fall_vs_hcm': {
            'group_A': ['FALL'],
            'group_B': ['HCM']
        },
        'CIA_vs_hcm': {
            'group_A': ['CIA'],
            'group_B': ['HCM']
        },
        'ARR_vs_hcm': {
            'group_A': ['ARR'],
            'group_B': ['HCM']
        },
        'dilated_like_vs_hcm': {
            'group_A': ['FALL','CIA','ARR'],
            'group_B': ['HCM']
        },
        'dilated_like_vs_normalish': {
            'group_A': ['FALL','CIA','ARR'],
            'group_B': ['NOR','HCM','DLV']
        },
    }

    SCANNER_HOLDOUTS = {
        'holdout_SymphonyTim': {'test_scanners': ['SymphonyTim']},
        'holdout_Achieva': {'test_scanners': ['Achieva']},
        'holdout_Signa_HDxt': {'test_scanners': ['Signa_HDxt']},
    }

    @staticmethod
    @lru_cache(maxsize=4)
    def build_manifest(data_dir: str) -> pd.DataFrame:
        data_dir = Path(data_dir).resolve()
        info_csv = data_dir / 'dataset_information.csv'
        if not info_csv.exists():
            raise FileNotFoundError(f'Missing metadata CSV at {info_csv}')
        meta = pd.read_csv(info_csv)
        if 'SUBJECT_CODE' not in meta.columns:
            meta = meta.reset_index().rename(columns={'index': 'SUBJECT_CODE'})
        rows = []
        root = data_dir / 'dataset'
        for _, r in meta.iterrows():
            sid = int(r['SUBJECT_CODE'])
            disease = r.get('DISEASE', None)
            scanner_raw = r.get('SCANNER', None)
            scanner_norm = MNMv2Dataset._SCANNER_RENAME.get(scanner_raw, scanner_raw)
            subj_dir = root / f"{sid:03d}"
            if not subj_dir.exists():
                continue
            for phase in ['ED','ES']:
                img = subj_dir / f"{sid:03d}_SA_{phase}.nii.gz"
                lbl = subj_dir / f"{sid:03d}_SA_{phase}_gt.nii.gz"
                if img.exists() and lbl.exists():
                    rows.append({
                        'SUBJECT_CODE': sid,
                        'DISEASE': disease,
                        'SCANNER': scanner_raw,
                        'SCANNER_NORM': scanner_norm,
                        'PHASE': phase,
                        'VIEW': 'SA',
                        'FILE_IMAGE': str(img),
                        'FILE_LABEL': str(lbl),
                    })
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError('Manifest is empty; verify dataset path/structure.')
        return df

    @staticmethod
    def split_scanner_holdout(manifest: pd.DataFrame, name: str, val_frac: float=0.1, seed: int=0):
        if name not in MNMv2Dataset.SCANNER_HOLDOUTS:
            raise ValueError(f"Unknown holdout '{name}'")
        test_scanners = MNMv2Dataset.SCANNER_HOLDOUTS[name]['test_scanners']
        test_subjects = set(manifest[manifest['SCANNER_NORM'].isin(test_scanners)]['SUBJECT_CODE'])
        train_subjects = list(set(manifest['SUBJECT_CODE']) - test_subjects)
        rng = random.Random(seed)
        rng.shuffle(train_subjects)
        n_val = int(round(val_frac * len(train_subjects)))
        val_subjects = set(train_subjects[:n_val])
        train_subjects = set(train_subjects[n_val:])
        return {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'phases_train': ['ED','ES'],
            'phases_val': ['ED','ES'],
            'phases_test': ['ED','ES'],
        }

    @staticmethod
    def split_pathology_exclusive(manifest: pd.DataFrame, protocol: str, val_frac: float=0.1, seed: int=0):
        if protocol not in MNMv2Dataset.PATHOLOGY_PROTOCOLS:
            raise ValueError(f"Unknown protocol '{protocol}'")
        spec = MNMv2Dataset.PATHOLOGY_PROTOCOLS[protocol]
        df = manifest[~manifest['DISEASE'].isin(MNMv2Dataset.EXCLUDE_PATHOLOGIES)]
        A = sorted(df[df['DISEASE'].isin(spec['group_A'])]['SUBJECT_CODE'].unique())
        B = set(df[df['DISEASE'].isin(spec['group_B'])]['SUBJECT_CODE'].unique())
        rng = random.Random(seed)
        rng.shuffle(A)
        n_val = int(round(val_frac * len(A)))
        val_subjects = set(A[:n_val])
        train_subjects = set(A[n_val:])
        return {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': B,
            'phases_train': ['ED','ES'],
            'phases_val': ['ED','ES'],
            'phases_test': ['ED','ES'],
        }

    @staticmethod
    def split_phase_exclusive(manifest: pd.DataFrame, val_frac: float=0.1, seed: int=0, require_both=True):
        """Subject-disjoint phase domain shift split (ED -> train/val; ES -> test).

        Previous implementation returned ALL subjects in test (overlapping) and was thus not
        truly "exclusive". This version enforces subject-level exclusivity:
          - Select a disjoint subset of subjects as the ES test set.
          - Remaining subjects contribute only ED phase to train/val.

        Parameters
        ----------
        manifest : pd.DataFrame
            Full MNMv2 manifest (from build_manifest).
        val_frac : float
            Fraction of remaining (ED) subjects used for validation.
        seed : int
            RNG seed for reproducibility.
        require_both : bool
            If True, only subjects with BOTH ED & ES are considered eligible.

        Returns
        -------
        dict with keys: train_subjects, val_subjects, test_subjects, phases_* lists.
        """
        pivot = manifest.groupby(['SUBJECT_CODE','PHASE']).size().unstack(fill_value=0)
        if require_both:
            eligible = pivot[(pivot.get('ED',0)>0) & (pivot.get('ES',0)>0)].index.tolist()
        else:
            eligible = pivot.index.tolist()
        if not eligible:
            raise ValueError('No eligible subjects (need both ED & ES when require_both=True).')
        rng = random.Random(seed)
        rng.shuffle(eligible)
        # Split subjects roughly in half: first half -> ES test only; second half -> ED train/val
        mid = max(1, len(eligible)//2)
        test_subjects = set(eligible[:mid])
        ed_pool = eligible[mid:]
        if not ed_pool:
            raise ValueError('Not enough subjects remaining for ED train/val after assigning test set.')
        rng.shuffle(ed_pool)
        n_val = int(round(val_frac * len(ed_pool))) if len(ed_pool) > 1 else 0
        val_subjects = set(ed_pool[:n_val]) if n_val>0 else set()
        train_subjects = set(ed_pool[n_val:]) if n_val>0 else set(ed_pool)
        # Sanity: disjointness
        assert len((train_subjects | val_subjects) & test_subjects) == 0, 'Phase split overlap detected.'
        return {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'phases_train': ['ED'],
            'phases_val': ['ED'],
            'phases_test': ['ES'],
        }

    @staticmethod
    def split_scanner_one_vs_all(manifest: pd.DataFrame, scanner: str, val_frac: float=0.1, seed: int=0, mode: str='source_train'):
        """Create a scanner domain shift split.

        mode:
          - 'holdout': selected scanner subjects go ONLY to test; all other scanners -> train/val (classic holdout)
          - 'source_train': selected scanner subjects form train/val; ALL other scanners -> test (requested one-vs-all source)
        Returns dict with train_subjects, val_subjects, test_subjects.
        """
        if 'SCANNER_NORM' not in manifest.columns:
            raise ValueError('Manifest missing SCANNER_NORM column')
        scanners = set(manifest.SCANNER_NORM.unique())
        if scanner not in scanners:
            raise ValueError(f"Scanner '{scanner}' not present. Available: {sorted(scanners)}")
        rng = random.Random(seed)
        if mode == 'holdout':
            test_subjects = set(manifest[manifest.SCANNER_NORM == scanner].SUBJECT_CODE.unique())
            pool = list(set(manifest.SUBJECT_CODE.unique()) - test_subjects)
            rng.shuffle(pool)
            n_val = int(round(val_frac * len(pool))) if pool else 0
            val_subjects = set(pool[:n_val])
            train_subjects = set(pool[n_val:])
        elif mode == 'source_train':
            source = list(manifest[manifest.SCANNER_NORM == scanner].SUBJECT_CODE.unique())
            if len(source) < 2:
                raise ValueError(f'Not enough subjects on scanner {scanner} for train/val split')
            rng.shuffle(source)
            n_val = max(1, int(round(val_frac * len(source))))
            val_subjects = set(source[:n_val])
            train_subjects = set(source[n_val:])
            test_subjects = set(manifest[manifest.SCANNER_NORM != scanner].SUBJECT_CODE.unique())
        else:
            raise ValueError("mode must be 'holdout' or 'source_train'")
        # If holdout, ensure test_subjects defined
        if mode == 'holdout':
            test_subjects = set(manifest[manifest.SCANNER_NORM == scanner].SUBJECT_CODE.unique())
        # disjoint check
        if len((train_subjects | val_subjects) & test_subjects) != 0:
            raise AssertionError('Overlap between train/val and test for scanner split')
        return {
            'mode': mode,
            'scanner': scanner,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'phases_train': ['ED','ES'],
            'phases_val': ['ED','ES'],
            'phases_test': ['ED','ES'],
        }

    @staticmethod
    def split_domain_shift(
        manifest: pd.DataFrame,
        axis: str,
        source_values,
        val_frac: float=0.1,
        seed: int=0,
        phase_mode: str='exclusive',  # retained for backward compatibility; only 'exclusive' supported now
        require_both_phases: bool=True,
        pathology_protocol: str=None,
    ):
        """Generic domain shift split builder.

        axis: 'scanner' | 'pathology' | 'phase'
        source_values: values defining the source domain (e.g. list of scanners OR ignored for pathology protocol OR list of scanners for phase restriction)
        phase_mode:
          - 'exclusive': ED -> train/val, ES -> test (subjects disjoint across phase usage)
          - 'shared': subjects present in both splits; ED samples into train/val, ES into test, but subject membership overlaps (domain shift on temporal phase only)
        pathology_protocol: if axis == 'pathology', name of protocol in PATHOLOGY_PROTOCOLS (group_A -> train/val, group_B -> test)
        Returns dict with subject id sets and metadata.
        """
        rng = random.Random(seed)
        if axis not in {'scanner','pathology','phase'}:
            raise ValueError("axis must be one of {'scanner','pathology','phase'}")
        if axis == 'scanner':
            if not source_values:
                raise ValueError('source_values required for scanner axis')
            src = set(source_values if isinstance(source_values,(list,set,tuple)) else [source_values])
            present = set(manifest.SCANNER_NORM.unique())
            missing = src - present
            if missing:
                raise ValueError(f'Source scanners missing in manifest: {sorted(missing)}')
            source_subjects = set(manifest[manifest.SCANNER_NORM.isin(src)].SUBJECT_CODE.unique())
            target_subjects = set(manifest[~manifest.SCANNER_NORM.isin(src)].SUBJECT_CODE.unique())
            pool = list(source_subjects)
            rng.shuffle(pool)
            n_val = max(1, int(round(val_frac * len(pool)))) if len(pool) > 1 else 0
            val_subjects = set(pool[:n_val]) if n_val>0 else set()
            train_subjects = set(pool[n_val:]) if n_val>0 else set(pool)
            return {
                'axis': axis,
                'source_values': sorted(src),
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subjects': target_subjects,
            }
        if axis == 'pathology':
            if not pathology_protocol:
                raise ValueError('pathology_protocol required for pathology axis')
            if pathology_protocol not in MNMv2Dataset.PATHOLOGY_PROTOCOLS:
                raise ValueError(f'Unknown pathology protocol {pathology_protocol}')
            spec = MNMv2Dataset.PATHOLOGY_PROTOCOLS[pathology_protocol]
            df = manifest[~manifest['DISEASE'].isin(MNMv2Dataset.EXCLUDE_PATHOLOGIES)]
            A = list(df[df.DISEASE.isin(spec['group_A'])].SUBJECT_CODE.unique())
            B = set(df[df.DISEASE.isin(spec['group_B'])].SUBJECT_CODE.unique())
            rng.shuffle(A)
            n_val = max(1, int(round(val_frac * len(A)))) if len(A) > 1 else 0
            val_subjects = set(A[:n_val]) if n_val>0 else set()
            train_subjects = set(A[n_val:]) if n_val>0 else set(A)
            return {
                'axis': axis,
                'protocol': pathology_protocol,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subjects': B,
            }
        if axis == 'phase':
            # Restrict manifest to specified scanners if provided
            if source_values:
                allowed_scanners = set(source_values if isinstance(source_values,(list,set,tuple)) else [source_values])
                manifest = manifest[manifest.SCANNER_NORM.isin(allowed_scanners)]
            pivot = manifest.groupby(['SUBJECT_CODE','PHASE']).size().unstack(fill_value=0)
            eligible = pivot[(pivot.get('ED',0)>0) & (pivot.get('ES',0)>0)].index.tolist()
            if not eligible:
                raise ValueError('No subjects with both ED and ES for phase split')
            rng.shuffle(eligible)
            if phase_mode != 'exclusive':
                raise ValueError("Only 'exclusive' phase_mode is supported now (subject-disjoint ED vs ES).")
            # Subject-disjoint exclusive: partition subjects into ES test vs ED train/val
            mid = max(1, len(eligible)//2)
            test_subjects = set(eligible[:mid])
            ed_pool = eligible[mid:]
            if not ed_pool:
                raise ValueError('Insufficient subjects remaining for ED train/val pool')
            rng.shuffle(ed_pool)
            n_val = max(1, int(round(val_frac * len(ed_pool)))) if len(ed_pool) > 1 else 0
            val_subjects = set(ed_pool[:n_val]) if n_val>0 else set()
            train_subjects = set(ed_pool[n_val:]) if n_val>0 else set(ed_pool)
            return {
                'axis': axis,
                'phase_mode': 'exclusive',
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subjects': test_subjects,
            }

    def __init__(
        self,
        data_dir,
        domain=None,
        binary_target: bool = False,
        non_empty_target: bool = True,
        normalize: bool = True,
        *,
        manifest: pd.DataFrame = None,
        subject_codes=None,
        phases=None,
        views=None,
    ):
        self.domain = domain
        self._data_dir = Path(data_dir).resolve()
        self._binary_target = binary_target
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._crop = CenterCrop(256)
        self.target_spacing = 1.

        if manifest is None:
            manifest = self.build_manifest(str(self._data_dir))
        self._manifest_full = manifest

        filt = manifest
        # Determine if advanced filtering is requested (avoid ambiguous truth values from numpy arrays)
        advanced = (subject_codes is not None) or (phases is not None) or (views is not None)
        if advanced:
            if subject_codes is not None:
                # accept numpy array / pandas series / list / set
                if hasattr(subject_codes, 'tolist'):
                    subject_codes_filter = set(subject_codes.tolist())
                else:
                    subject_codes_filter = set(list(subject_codes))
                filt = filt[filt['SUBJECT_CODE'].isin(subject_codes_filter)]
            if phases is not None:
                phases_filter = list(phases)
                filt = filt[filt['PHASE'].isin(phases_filter)]
            if views is not None:
                views_filter = list(views)
                filt = filt[filt['VIEW'].isin(views_filter)]
        else:
            if domain is None:
                raise ValueError("Provide `domain` for legacy mode or filters for advanced mode.")
            filt = filt[filt['SCANNER'].str.lower().str.contains(domain.lower())]

        if filt.empty:
            raise ValueError('Filtered manifest is empty. Check filters or domain input.')
        self._manifest = filt.reset_index(drop=True)
        self._load_from_manifest()

    def _load_from_manifest(self):
        self.input = []
        self.target = []
        for _, row in self._manifest.iterrows():
            x_img = nib.load(row['FILE_IMAGE'])
            y_img = nib.load(row['FILE_LABEL'])
            spacing = x_img.header.get_zooms()[0]
            x = torch.tensor(x_img.get_fdata()).moveaxis(-1, 0)
            y = torch.tensor(y_img.get_fdata().astype(int), dtype=torch.long).moveaxis(-1, 0)
            scale_factor = (1 / self.target_spacing * spacing, 1 / self.target_spacing * spacing)
            x = interpolate(x.unsqueeze(1), scale_factor=scale_factor, mode='bilinear', align_corners=True).squeeze(1)
            y = interpolate(y.unsqueeze(1).float(), scale_factor=scale_factor, mode='nearest').long().squeeze(1)
            x = self._crop(x)
            y = self._crop(y)
            self.input.append(x)
            self.target.append(y)

        self.input = torch.cat(self.input, dim=0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=0).unsqueeze(1)
        self.target[self.target < 0] = 0
        if self._non_empty_target:
            mask = self.target.sum((-1,-2,-3)) > 0
            self.input = self.input[mask]
            self.target = self.target[mask]
        if self._binary_target:
            self.target[self.target != 0] = 1
        if self._normalize:
            mean = self.input.mean(); std = self.input.std()
            self.input = (self.input - mean) / (std + 1e-8)


    def random_split(
        self,
        val_size: float = 0.2,
        test_size: float = None,
    ):
        class MNMv2Subset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]
            
            def __getitem__(self, idx):
                return {
                    "input": self.input[idx], 
                    "target": self.target[idx],
                    "index": idx
                }
        

        torch.manual_seed(0)
        indices = torch.randperm(len(self.input)).tolist()

        if test_size is not None:

            test_split = int(test_size * len(self.input))
            val_split = int(val_size * len(self.input)) + test_split

            mnmv2_test = MNMv2Subset(
                input=self.input[indices[:test_split]],
                target=self.target[indices[:test_split]],
            )

            mnmv2_val = MNMv2Subset(
                input=self.input[indices[test_split:val_split]],
                target=self.target[indices[test_split:val_split]],
            )

            mnmv2_train = MNMv2Subset(
                input=self.input[indices[val_split:]],
                target=self.target[indices[val_split:]],
            )


            return mnmv2_train, mnmv2_val, mnmv2_test
        
        mnmv2_train = MNMv2Subset(
            input=self.input[indices[int(val_size * len(self.input)):]],
            target=self.target[indices[int(val_size * len(self.input)):]],
        )

        mnmv2_val = MNMv2Subset(
            input=self.input[indices[:int(val_size * len(self.input))]],
            target=self.target[indices[:int(val_size * len(self.input))]],
        )

        return mnmv2_train, mnmv2_val


    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "index": idx,
        }
