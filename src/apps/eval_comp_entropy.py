import sys, os
from omegaconf import OmegaConf
import hydra
import numpy as np
from torch import (
    stack,
    cat,
    corrcoef,
    zeros,
    save,
    manual_seed,
    tensor,
    sqrt,
)
from torch.nn.functional import one_hot, softmax
from torch.utils.data import DataLoader
from scipy.stats import entropy as entr
from tqdm import tqdm

sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from losses import dice_per_class_loss, surface_loss

@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):
    
    eval_metrics = {
        'dice': dice_per_class_loss,
        # 'surface': surface_loss  # Keep for later
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

    unet = get_unet_module(
        cfg=unet_cfg,
        metadata=OmegaConf.to_container(unet_cfg),
        load_from_checkpoint=True
    ).model

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
            'entropy': []
        }

        manual_seed(cfg.run_id)
        
        for batch in tqdm(test_dl):
            input = batch['input']
            target = batch['target']

            # For single sample (batch_size=1): 
            #   - Dropout deactivated → deterministic predictions
            #   - Entropy comes from the probability distribution itself (sharpness)
            if cfg.batch_size == 1:
                unet.eval()  # Deactivates all dropout layers
                logits = unet(input.cuda())
                logits_dropout = logits
            else:
                # For multiple samples (batch_size>1):
                #   - Dropout activated → stochastic predictions
                #   - Probabilities averaged across dropout samples
                #   - Entropy comes from uncertainty in the averaged distribution
                input_repeated = input.repeat(cfg.batch_size, 1, 1, 1)
                
                unet.eval()
                logits = unet(input.cuda())
                # Activate dropout layers for Monte Carlo sampling
                for m in unet.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()

                logits_dropout = unet(input_repeated.cuda())

            num_classes = max(logits_dropout.shape[1], 2)
            if num_classes > 2:
                predictions = logits.argmax(1, keepdim=True)
                # Average predictions across dropout runs and compute probabilities
                probs = softmax(logits_dropout, dim=1).mean(0, keepdim=True).detach()
            else:
                predictions = (logits > 0) * 1
                # For binary, convert logits to probabilities
                probs = softmax(cat([logits_dropout, -logits_dropout], dim=1), dim=1).mean(0, keepdim=True).detach()

            # Calculate predictive entropy
            # probs shape: [1, C, H, W]
            # scipy.stats.entropy computes along axis=0 by default
            # We need to compute entropy along the class dimension (axis=1)
            probs_np = probs.cpu().numpy()
            
            # Compute entropy per pixel along class dimension
            # entr() with axis=1 will give shape [1, H, W]
            entr_values = entr(probs_np, axis=1)
            
            # Average entropy across spatial dimensions
            entropy_mean = entr_values.mean()
            
            # Normalize by max entropy (log(num_classes)) and invert so higher is more certain
            max_entropy = np.log(num_classes)
            entropy = 1 - (entropy_mean / max_entropy) if max_entropy > 0 else 0.0
            
            scores['entropy'].append(tensor([entropy]).view(1,))

            fn = eval_metrics[score_type]
            _, _, true_score = fn(
                predicted_segmentation=predictions, 
                target_segmentation=target.cuda(),
                prediction=zeros((input.size(0), 1, num_classes)).cuda(),
                num_classes=num_classes,
                return_scores=True
            )
            scores[score_type].append(true_score.squeeze(1).detach().cpu())

        scores = {key: cat(scores[key]) for key in scores.keys()}

        correlation = corrcoef(stack([scores['entropy'], scores[score_type]], dim=0))[0,1]
        
        results[test_domain] = {
            'scores': scores,
            'correlation': correlation.item()
        }

        print(f"Correlation entropy-{score_type}: {correlation:.4f}")

    save_path = f'../../results/{dataset}_{split}_{score_type}_comp-entropy-{cfg.batch_size}-{cfg.run_id}.pt'
    save(results, save_path)
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    main()
