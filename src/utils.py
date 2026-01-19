from __future__ import annotations
import sys
import random
from datetime import datetime
from pathlib import Path
import re
from omegaconf import OmegaConf
from typing import (
    Optional,
    Union,
    Dict, 
    Tuple, 
    List
)
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader



_ALLOWED = re.compile(r"[^a-z0-9]+")


def eAURC(predicted_risks, true_risks, ret_curves=False):
    n = len(true_risks)

    true_risks_sorted, _ = true_risks.sort()
    _, predicted_indices = predicted_risks.sort()

    true_risks_aggr = true_risks_sorted.cumsum(0) / torch.arange(1, n + 1)
    aurc_opt = true_risks_aggr.mean()

    predicted_risks_aggr = true_risks[predicted_indices].cumsum(0) / torch.arange(1, n + 1)
    aurc_pred = predicted_risks_aggr.mean()

    assert true_risks_aggr[-1] == predicted_risks_aggr[-1]

    if ret_curves:
        return aurc_pred - aurc_opt, true_risks_aggr, predicted_risks_aggr
    else:
        return aurc_pred - aurc_opt
    
    
def reject_randomness(manualSeed):
    np.random.seed(manualSeed)
    random.rand.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    
    return weighted_losses.sum()/counts_np.sum()



def dataset_from_indices(dataset: Dataset, indices: Tensor) -> DataLoader:
    data = dataset.__getitem__(indices)
    
    class CustomDataset(Dataset):
        
        def __init__(self, input: Tensor, labels: Tensor, 
                     voxel_dim: Tensor):
            self.input = input
            self.labels = labels
            self.voxel_dim = voxel_dim
            
        def __getitem__(self, idx):
            return {'input': self.input[idx],
                    'target': self.labels[idx],
                    'voxel_dim': self.voxel_dim[idx]}
        
        def __len__(self):
            return self.input.size(0)
        
    return CustomDataset(*data.values())


 
def load_state_dict_for_modulelists(model, state_dict):

    seg_model_dict = model.seg_model.state_dict()
    seg_model_dict_pretrained = {
        k.replace('seg_model.', ''): v for k, v in state_dict.items() if k.replace('seg_model.', '') in seg_model_dict
    }
    model.seg_model.load_state_dict(seg_model_dict_pretrained)

    counter = 0
    for i in range(4):
        transformation_state_dict = model.transformations[i].state_dict()
        for j in range(4):
            try:
                transformation_state_dict_pretrained = {
                    k.replace(f'transformations.{j}.', ''): v for k, v in state_dict.items() if k.replace(f'transformations.{j}.', '') in transformation_state_dict
                }
                model.transformations[i].load_state_dict(transformation_state_dict_pretrained)
                counter += 1
            except:
                pass
    if counter == 4:
        print('All transformations loaded')
    else:
        sys.exit()

    return model



def sum_model_parameters(model: nn.Module) -> int:
    """
    Sum up the number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def find_shapes_for_swivels(
    model: nn.Module, 
    swivels: List[str],
    input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256)
) -> Dict[str, List[int]]:
    # Create a dictionary to store the output shapes
    model = deepcopy(model).cuda()
    output_shapes = {}

    # Get hook function to capture and print output shapes for swivel
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            output_shapes[name] = list(output.shape)
        return hook_fn
    # def get_hook_fn(name):
    #     def hook_fn(module, input):
    #         output_shapes[name] = list(input[0].shape)
    #     return hook_fn

    # Attach hooks to all layers
    hooks = []
    for layer_id in swivels:
        layer   = model.get_submodule(layer_id)
        hook_fn = get_hook_fn(layer_id)
        hook    = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Run a sample input through the model
    x = torch.randn(input_shape).cuda()  # Batch size 1, 3 channels, 32x32 image
    _ = model(x)

    # remove hooks and model
    for hook in hooks:
        hook.remove()

    del model

    return output_shapes


def sanitize_tag(s: str) -> str:
    """Lowercase; replace any non-alnum with '-'; collapse repeats; strip ends."""
    s = str(s).lower()
    s = _ALLOWED.sub('-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s


def _format_dropout_tag(dropout: Optional[Union[int, float, str]]) -> str:
    """Two-digit tenths precision: 0.0->00, …, 0.9->09, 1.0->10; 'na' if not provided."""
    if dropout is None or dropout == '':
        return 'na'
    try:
        v = float(dropout)
    except Exception:
        return sanitize_tag(str(dropout))
    code = int(round(v * 10))
    code = max(0, min(code, 10))
    return f'{code:02d}'


def scenario_to_split_tag(scenario: str) -> str:
    """
    MNMv2 scenario -> split tag with hyphens.
    Examples:
      'scanner:source=SymphonyTim' -> 'scanner-symphonytim'
      'pathology:protocol=dilated_like_vs_hcm:scanners=ALL'
         -> 'pathology-dilated-like-vs-hcm-scanners-all'
      'phase:mode=exclusive:scanners=SymphonyTim'
         -> 'phase-exclusive-scanners-symphonytim'
    """
    parts = str(scenario).split(':')
    if not parts:
        return sanitize_tag(scenario)
    axis = sanitize_tag(parts[0])
    tokens: List[str] = [axis]
    for p in parts[1:]:
        if '=' not in p:
            tokens.append(sanitize_tag(p))
            continue
        k, v = p.split('=', 1)
        k = sanitize_tag(k)
        v = sanitize_tag(v.replace('_', '-'))
        if k in ('source', 'protocol', 'mode'):
            tokens.append(v)
        elif k == 'scanners':
            tokens.extend([k, v])
        else:
            tokens.extend([k, v])
    return sanitize_tag('-'.join(tokens))


def _compute_split_tag(data_cfg: OmegaConf) -> str:
    dataset = sanitize_tag(getattr(data_cfg, 'dataset', 'unknown'))
    # Prefer explicit split tag universally (MNMv2 and PMRI both use canonical split keys now)
    split = getattr(data_cfg, 'split', None)
    if split:
        return sanitize_tag(split)
    domain = getattr(data_cfg, 'domain', None)
    if domain:
        return sanitize_tag(domain)
    return 'unknown'


def build_checkpoint_filename(data_cfg: OmegaConf, unet_cfg: OmegaConf, now: Optional[datetime] = None) -> str:
    dataset_tag = sanitize_tag(getattr(data_cfg, 'dataset', 'unknown'))
    split_tag = _compute_split_tag(data_cfg)
    dropout_tag = _format_dropout_tag(getattr(unet_cfg, 'dropout', None))
    ts = (now or datetime.now()).strftime('%Y-%m-%d-%H-%M')
    return f'{dataset_tag}_{split_tag}_dropout-{dropout_tag}_{ts}'


def find_unet_checkpoint(
    base_dir: Union[str, Path],
    *,
    dataset: str,
    split: str,
    dropout: Optional[Union[int, float, str]] = None,
    newest: bool = True
):
    """
    Return newest checkpoint path matching dataset/split (+optional dropout).
    Pattern: dataset_split_dropout-XX_YYYY-MM-DD-HH-MM.ckpt
    """
    base = Path(base_dir)
    dataset_tag = sanitize_tag(dataset)
    split_tag = sanitize_tag(split)
    dtag = _format_dropout_tag(dropout) if dropout is not None else '*'
    pattern = f'{dataset_tag}_{split_tag}_dropout-{dtag}_*.ckpt'
    candidates = list(base.glob(pattern))
    if not candidates:
        return None

    def ts_key(p: Path):
        m = re.search(r'_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.ckpt$', p.name)
        if m:
            try:
                return datetime.strptime(m.group(1), '%Y-%m-%d-%H-%M')
            except Exception:
                pass
        return datetime.fromtimestamp(p.stat().st_mtime)

    candidates.sort(key=ts_key, reverse=True)
    return candidates[0] if newest else candidates
