import sys
from omegaconf import OmegaConf
import hydra
sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module, get_unet_module_trainer
from utils import find_unet_checkpoint
import torch
import lightning as L
from pathlib import Path



@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):
    # Log context: support PMRI or MNMv2 via canonical split tags
    data_cfg = cfg.data
    split = getattr(data_cfg, 'split', None)
    if data_cfg.dataset == 'pmri':
        if split is None:
            raise ValueError("[ERROR] PMRI requires a predefined split tag in data.split")
        print(f"[INFO] Starting training for PMRI split='{split}'")
    elif data_cfg.dataset == 'mnmv2':
        if split is not None:
            print(f"[INFO] Starting training for MNMv2 split='{split}'")
        else:
            raise ValueError("[ERROR] MNMv2 requires a predefined split tag in data.split")
    else:
        print(f"[INFO] Starting training for dataset='{data_cfg.dataset}'")

    # init datamodule
    datamodule = get_data_module(
        cfg=cfg.data
    )

    # init model (load existing checkpoint if available)
    ckpt_dir = Path(cfg.trainer.model_checkpoint.dirpath)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (Path(__file__).resolve().parent / ckpt_dir).resolve()

    ckpt_path = None
    if split is not None:
        ckpt_path = find_unet_checkpoint(
            ckpt_dir,
            dataset=data_cfg.dataset,
            split=split,
            dropout=getattr(cfg.unet, 'dropout', None),
            newest=True
        )

    if ckpt_path is not None:
        cfg.unet.checkpoint_path = str(ckpt_path)
        load_from_checkpoint = True
        print(f"[INFO] Found checkpoint: {ckpt_path}")
    else:
        load_from_checkpoint = False
        print(f"[INFO] No checkpoint found in {ckpt_dir}; training from scratch")

    model = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg),
        load_from_checkpoint=load_from_checkpoint
    )

    # init trainer
    trainer = get_unet_module_trainer(
        data_cfg=cfg.data,
        model_cfg=cfg.unet,
        trainer_cfg=cfg.trainer
    )

    # train
    if not load_from_checkpoint:
        trainer.fit(model=model, datamodule=datamodule)


    datamodule.setup('test')
    test_loaders = datamodule.test_dataloader()
    # Ensure we can iterate: expected dict from PMRIDataModule / others
    if isinstance(test_loaders, dict):
        loader_keys = list(test_loaders.keys())
        loader_list = [test_loaders[k] for k in loader_keys]
        predictions = trainer.predict(model, loader_list)
        # Aggregate metrics assuming each item in predictions is a list of dicts
        metrics_clean = {}
        for key, pred_batches in zip(loader_keys, predictions):
            # pred_batches: list[dict[str, Tensor]]
            if not pred_batches:
                continue
            aggregated = {}
            for k in pred_batches[0].keys():
                try:
                    aggregated[k] = torch.cat([b[k] for b in pred_batches if k in b], dim=0)
                except Exception:
                    # Fallback: store list if cat not possible
                    aggregated[k] = [b[k] for b in pred_batches if k in b]
            metrics_clean[key] = aggregated
    else:
        predictions = trainer.predict(model, test_loaders)
        # predictions: list of dicts -> consolidate
        metrics_clean = {}
        if predictions:
            for k in predictions[0].keys():
                try:
                    metrics_clean[k] = torch.cat([b[k] for b in predictions if k in b], dim=0)
                except Exception:
                    metrics_clean[k] = [b[k] for b in predictions if k in b]

    # Save metrics
    results_dir = Path('../../results/unet_eval_auto')
    results_dir.mkdir(parents=True, exist_ok=True)
    # Name file by dataset + split
    tag = split
    outfile = results_dir / f"{data_cfg.dataset}_{tag}.pt"
    torch.save(metrics_clean, outfile)
    print(f"[INFO] Saved evaluation metrics to {outfile}")


if __name__ == '__main__':
    main()