import os
import sys 
sys.path.append('..')

import argparse
import torch
import timm
import pytorch_lightning as pl
from omegaconf import OmegaConf
import json
from pathlib import Path
from data import CustomDataModule
from engine import BinaryEngine, MultiEngine
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, progress, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor



def create_ckpt_callback(ckpt_config):
    return ModelCheckpoint(
        filename='{epoch}-{val_acc:.2f}-{val_bce_loss:.2f}',
        save_top_k=ckpt_config.save_top_k,
        monitor=ckpt_config.monitor,
        mode=ckpt_config.mode,
    )

    
def create_trainer(config):

    tb_logger = loggers.TestTubeLogger(config.experiment.save_dir, name=config.experiment.name)
    tb_logger.log_hyperparams(config)

    early_stop_callback = EarlyStopping(
        monitor=config.experiment.early_stop.monitor,
        min_delta=0.00,
        patience=config.experiment.early_stop.patience,
        verbose=True,
        mode=config.experiment.early_stop.mode
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    _callbacks = [progress.ProgressBar(), create_ckpt_callback(config.experiment.ckpt_callback), early_stop_callback, lr_monitor]

    trainer = pl.Trainer(logger=tb_logger,
                         gpus=config.experiment.gpu if config.experiment.accelerator != 'cpu' else None,
                         max_epochs=config.training.max_epochs,
                         accelerator=config.experiment.accelerator,
                         progress_bar_refresh_rate=10,
                         deterministic=False,
                         terminate_on_nan=True,
                         num_sanity_val_steps=10,
                         callbacks=_callbacks,
                         accumulate_grad_batches=config.training.grad_acc
                         )
    return trainer


def train(config):

    pl.seed_everything(config.experiment.seed)

    data_conf = config.dataset.fgvcx
    data_handler = CustomDataModule(data_conf)
    data_handler.setup()

    model = timm.create_model(
        config.model.arch, 
        num_classes=0, 
        drop_rate=config.model.dropout, 
        pretrained=config.model.pretrained
    )

    if config.model.ckpt is not None:
        ckp = torch.load(config.model.ckpt, map_location='cpu')

        state = {k.replace('student.', ''): v for k, v in ckp['state_dict'].items() if 'student' in k}

        model.load_state_dict(state, strict=False)
    
    model.reset_classifier(num_classes=config.model.n_classes)
    
    if config.training.train_fc_only:
        for name, child in model.named_children():
            if name not in config.model.unfreeze_modules:
                for param in child.parameters():
                    param.requires_grad = False
    
    engine = MultiEngine(
        student=model,
        criterion=config.training.loss,
        opt=config.training.opt,
        lr=config.training.lr,
        wd=config.training.weight_decay,
        scheduler_conf=config.training.scheduler
        )
    
    trainer = create_trainer(config)

    trainer.fit(
        model=engine, 
        train_dataloader=data_handler.train_dataloader(), 
        val_dataloaders=data_handler.val_dataloader()
    )

    best = trainer.checkpoint_callback.best_model_path
    
    print("Test the model against train set:")
    trainer.test(ckpt_path=best, dataloaders=[data_handler.train_dataloader()])
    print("Test the model against test set:")
    test_result = trainer.test(ckpt_path=best, dataloaders=[data_handler.test_dataloader()])
    
    test_acc = test_result[0]['test_acc']

    with open(Path(best).parent.parent / f'{Path(best).stem}_test_acc_{test_acc:.2f}.json', 'w') as j:
        json.dump({'test': test_acc}, j)



if __name__ == '__main__':

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Provided config not found - {args.config_path}.')

    config = OmegaConf.load(args.config)

    train(config)