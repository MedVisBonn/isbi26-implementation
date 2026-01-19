import sys, os, glob
import pickle as pkl
from omegaconf import OmegaConf
import hydra
sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from model.score_adapter import (
    get_score_prediction_module, 
    get_score_prediction_module_trainer,
    clean_predictions,
    collect_eval_from_predictions,
)



@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):
    
    # init datamodule
    datamodule = get_data_module(
        cfg=cfg.data
    )

    # init model
    # Resolve UNet checkpoint by dataset + split (+optional dropout)
    base_dir = os.path.join(cfg.trainer.root_dir, cfg.unet.checkpoint_dir)
    ckpt_path = find_unet_checkpoint(
        base_dir,
        dataset=cfg.data.dataset,
        split=cfg.data.split,
        dropout=getattr(cfg.unet, 'dropout', None),
        newest=True
    )
    if ckpt_path is None:
        raise FileNotFoundError(f"No UNet checkpoint found in {base_dir} for dataset={cfg.data.dataset} split={cfg.data.split}")
    cfg.unet.checkpoint_path = str(ckpt_path)
    unet = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg.unet),
        load_from_checkpoint=True
    ).model

    if cfg.train:
        ckpt = None  # will be determined after fit
    else:
        ckpt = cfg.trainer.root_dir + cfg.trainer.ckpt_dir + cfg.trainer.ckpt_name
        assert os.path.exists(ckpt), f'Checkpoint not found at {ckpt}'

    print(OmegaConf.to_yaml(cfg))
    
    # cfg.model.hausdorff_sigma = HAUSDORFF_SIGMAS[cfg.data.dataset][cfg.data.split]
    model = get_score_prediction_module(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        unet=unet,
        metadata=OmegaConf.to_container(cfg.model), #TODO
        ckpt=ckpt
    )

    # init trainer
    trainer = get_score_prediction_module_trainer(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        trainer_cfg=cfg.trainer
    )
    
    # train
    if cfg.train:
        trainer.fit(model=model, datamodule=datamodule)
        # Best val checkpoint (first checkpoint callback)
        ckpt = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        # Locate train-best checkpoint if enabled
        train_best_ckpt = None
        if getattr(cfg.trainer, 'enable_train_loss_ckpt', False):
            ckpt_search_dir = os.path.join(cfg.trainer.root_dir, cfg.trainer.model_checkpoint.dirpath)
            pattern = os.path.join(ckpt_search_dir, f"*{cfg.model.name}*train-*.ckpt")
            matches = glob.glob(pattern)
            if matches:
                # Expect only one (save_top_k=1); take first
                train_best_ckpt = matches[0]
        else:
            train_best_ckpt = None
    else:
        train_best_ckpt = None

    # test
    if cfg.test:
        # Always evaluate best val (ckpt) first
        eval_ckpts = []
        if ckpt is not None and os.path.exists(ckpt):
            eval_ckpts.append(('val', ckpt))
        if train_best_ckpt is not None and os.path.exists(train_best_ckpt):
            eval_ckpts.append(('train', train_best_ckpt))

        for tag, ckpt_path in eval_ckpts:
            prediction = trainer.predict(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path
            )
            prediction_clean = clean_predictions(prediction, datamodule, model)
            evaluation = collect_eval_from_predictions(prediction_clean)
            model_name = ckpt_path.split('/')[-1].replace('0.', '0-').split('.')[0]
            save_dir = f'{cfg.trainer.root_dir}/results/{cfg.trainer.result_dir}/{model_name}'
            os.makedirs(save_dir, exist_ok=True)
            packaged = {
                'metadata': {
                    'data': OmegaConf.to_container(cfg.data),
                    'unet': OmegaConf.to_container(cfg.unet),
                    'model': OmegaConf.to_container(cfg.model),
                    'trainer': OmegaConf.to_container(cfg.trainer),
                    'checkpoint_type': tag
                },
                'evaluation': evaluation
            }
            with open(f'{save_dir}/eval_data.pkl', 'wb') as f:
                pkl.dump(packaged, f)



if __name__ == '__main__':
    main()