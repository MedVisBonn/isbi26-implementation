import sys, os
from omegaconf import OmegaConf
import hydra
import numpy as np
import torch
from torch import (
    stack,
    cat,
    corrcoef,
    zeros,
    save,
    manual_seed,
    tensor,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from model.mahalanobis import get_pooling_mahalanobis_detector
from losses import dice_per_class_loss

@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):
    
    eval_metrics = {
        'dice': dice_per_class_loss,
    }
    
    score_type = cfg.score_type

    unet_cfg = OmegaConf.load('../configs/unet/monai_unet.yaml')

    dataset = cfg.data.dataset
    split = cfg.data.split
    if dataset == 'mnmv2':
        unet_cfg.out_channels = 4
        num_classes = 4
    else:
        unet_cfg.out_channels = 1
        num_classes = 2

    data_cfg = cfg.data
    data_cfg.non_empty_target = True

    datamodule = get_data_module(cfg=data_cfg)
    datamodule.setup('test')

    # Resolve UNet checkpoint by dataset + split (+optional dropout)
    base_dir = os.path.join('..', '..', unet_cfg.checkpoint_dir)
    ckpt_path = find_unet_checkpoint(
        base_dir,
        dataset=dataset,
        split=split,
        dropout=getattr(unet_cfg, 'dropout', None),
        newest=True
    )
    if ckpt_path is None:
        raise FileNotFoundError(f"No UNet checkpoint found in {base_dir} for dataset={dataset} split={split}")
    unet_cfg.checkpoint_path = str(ckpt_path)

    module = get_unet_module(
        cfg=unet_cfg,
        metadata=OmegaConf.to_container(unet_cfg),
        load_from_checkpoint=True
    )
    unet = module.model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    unet = unet.to(device)
    unet.eval()

    # Determine swivel(s) via discovery or use provided list
    local_swivels = getattr(cfg, 'swivels', None)
    if local_swivels is None and cfg.get('discover_swivel', True):
        swivel_match = cfg.get('swivel_match', 'adn.N')
        swivel_indices = cfg.get('swivel_indices', [17])
        
        matched = [name for (name, _m) in unet.named_modules() if swivel_match in name]
        if not matched:
            print(f"[warn] No modules matched '{swivel_match}'. Using root hook.")
            local_swivels = ['']
        else:
            sel = []
            for idx in swivel_indices:
                if -len(matched) <= idx < len(matched):
                    sel.append(matched[idx])
            if not sel:
                print(f"[warn] Provided indices {swivel_indices} out of range for {len(matched)} matches. Using last match.")
                sel = [matched[-1]]
            local_swivels = sel
        print(f"[info] Using swivels: {local_swivels}")
    elif local_swivels is None:
        local_swivels = ['']

    # Configuration for Mahalanobis detector
    pool = cfg.get('pool', 'avg2d')
    sigma_algorithm = cfg.get('sigma_algorithm', 'ledoit_wolf')
    dist_fn = cfg.get('dist_fn', 'squared_mahalanobis')
    
    # Extract swivel info for filename
    swivel_id_str = '_'.join([str(idx) for idx in cfg.get('swivel_indices', [17])])

    # 1) FIT MAHALANOBIS on training data
    dataset_train = datamodule.dataset_train
    train_loader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=False,
    )
    train_iter = (batch for batch in tqdm(train_loader, desc=f"Fitting Mahalanobis ({dataset}/{split})", leave=False))
    
    pooling = get_pooling_mahalanobis_detector(
        swivels=local_swivels,
        unet=unet,
        pool=pool,
        sigma_algorithm=sigma_algorithm,
        dist_fn=dist_fn,
        fit='raw',
        iid_data=train_iter,
        transform=False,
        device=device_str
    )

    # 2) EVALUATE on test domains
    results = {}

    for test_domain, test_dl in datamodule.test_dataloader().items():
        if not cfg.include_train and ('train' in test_domain):
            continue

        print(f"Evaluating domain: {test_domain}")
        
        test_dataset = test_dl.dataset
        test_dl = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
        )

        scores = {
            score_type: [],
            'mahalanobis': []
        }

        manual_seed(cfg.run_id)
        
        for batch in tqdm(test_dl, desc=f"Evaluating {test_domain}"):
            input = batch['input'].to(device)
            target = batch['target'].to(device)

            with torch.no_grad():
                # Trigger Mahalanobis adapters
                _ = pooling(input)
                
                # Collect distances from adapters and flip sign
                # (lower distance = closer to training = better quality)
                batch_vals = []
                for adapter in pooling.adapters:
                    batch_vals.append(adapter.batch_distances.detach().cpu())
                mean_dist = torch.stack(batch_vals, dim=0).mean(0)
                # Flip: -distance so higher values indicate better quality
                scores['mahalanobis'].extend((-mean_dist).tolist())

                # Compute Dice score
                logits = unet(input)
                if num_classes > 2:
                    predictions = logits.argmax(1, keepdim=True)
                else:
                    predictions = (logits > 0).long()

                fn = eval_metrics[score_type]
                _, _, true_score = fn(
                    predicted_segmentation=predictions.cpu(),
                    target_segmentation=target.cpu(),
                    prediction=zeros((input.size(0), 1, num_classes)),
                    num_classes=num_classes,
                    return_scores=True
                )
                # Ensure we keep at least 1D (squeeze(1) instead of squeeze())
                scores[score_type].append(true_score.squeeze(1).detach().cpu())

        # Convert lists to tensors
        scores['mahalanobis'] = tensor(scores['mahalanobis'])
        scores[score_type] = cat(scores[score_type])

        # Compute correlation
        if len(scores['mahalanobis']) > 1 and len(scores[score_type]) > 1:
            correlation = corrcoef(stack([scores['mahalanobis'], scores[score_type]], dim=0))[0,1]
            results[test_domain] = {
                'scores': scores,
                'correlation': correlation.item()
            }
            print(f"Correlation mahalanobis-{score_type}: {correlation:.4f}")
        else:
            results[test_domain] = {
                'scores': scores,
                'correlation': None
            }
            print(f"Insufficient samples for correlation in {test_domain}")

    save_path = f'../../results/{dataset}_{split}_{score_type}_mahalanobis-{sigma_algorithm}-swivel{swivel_id_str}-{cfg.run_id}.pt'
    save(results, save_path)
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    main()
