import sys, os
from omegaconf import OmegaConf
import hydra
from torch import (
    stack,
    cat,
    corrcoef,
    zeros,
    save,
    manual_seed,
    triu_indices,
)
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, SurfaceDiceMetric
from tqdm import tqdm

sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from losses import dice_per_class_loss, surface_loss

def pairwise_dice(predicted_segmentation, num_classes):
    predicted_segmentation = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
    N = predicted_segmentation.shape[0]
    i_idx, j_idx = triu_indices(N, N, offset=1)

    pred = predicted_segmentation[i_idx]
    ref = predicted_segmentation[j_idx]

    dice_scores = DiceMetric(
        include_background=True, 
        reduction="none",
        num_classes=num_classes,
        ignore_empty=False
    )(pred, ref)[..., 1:].nanmean(-1).nan_to_num(0).cpu().detach().mean()

    return dice_scores

def pairwise_surface_dice(predicted_segmentation, num_classes):
    predicted_segmentation = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
    N = predicted_segmentation.shape[0]
    i_idx, j_idx = triu_indices(N, N, offset=1)

    pred = predicted_segmentation[i_idx]
    ref = predicted_segmentation[j_idx]

    surface_scores = SurfaceDiceMetric(
        include_background=True, 
        reduction="none",
        class_thresholds=[3] * num_classes,
    )(pred, ref).detach()[..., 1:].nanmean(-1).nan_to_num(0).cpu().detach().mean()

    return surface_scores

@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):
    
    eval_metrics = {
        'dice': dice_per_class_loss,
        'surface': surface_loss
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
            f'{score_type}_agreement': []
        }

        manual_seed(cfg.run_id)
        
        for batch in tqdm(test_dl):
            input = batch['input'].repeat(cfg.batch_size, 1, 1, 1)
            target = batch['target']

            unet.eval()
            logits = unet(input[:1].cuda())
            for m in unet.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

            logits_dropout = unet(input.cuda())

            num_classes = max(logits_dropout.shape[1], 2)
            if num_classes > 2:
                predictions = logits.argmax(1, keepdim=True)
                predictions_dropout = logits_dropout.argmax(1, keepdim=True)
            else:
                predictions = (logits > 0) * 1
                predictions_dropout = (logits_dropout > 0) * 1

            if score_type == 'dice':
                agreement = pairwise_dice(predictions_dropout, num_classes=num_classes)
            else:
                agreement = pairwise_surface_dice(predictions_dropout, num_classes=num_classes)
            
            scores[f'{score_type}_agreement'].append(agreement.detach().cpu().view(1,))

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

        correlation = corrcoef(stack([scores[f'{score_type}_agreement'], scores[score_type]], dim=0))[0,1]
        
        results[test_domain] = {
            'scores': scores,
            'correlation': correlation.item()
        }

        print(f"Correlation {score_type}: {correlation:.4f}")

    save_path = f'../../results/{dataset}_{split}_{score_type}_score-agreement-{cfg.batch_size}-{cfg.run_id}.pt'
    save(results, save_path)
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    main()