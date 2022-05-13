import os
import sys

from sklearn.metrics import classification_report 
sys.path.append('..')

import argparse
import torch
import timm
import pytorch_lightning as pl
from omegaconf import OmegaConf
from data import CustomDataModule


def eval(config, ckpt_path):

    pl.seed_everything(config.experiment.seed)

    data_conf = config.dataset.fgvcx
    data_handler = CustomDataModule(data_conf)
    data_handler.setup()

    model = timm.create_model(
        config.model.arch,
        num_classes=config.model.n_classes,
        pretrained=False
    )

    ckp = torch.load(ckpt_path, map_location='cpu')
    state = {k.replace('student.', ''): v for k, v in ckp['state_dict'].items() if 'student' in k}
    model.load_state_dict(state, strict=False)
    
    model.eval()
    model.to(config.experiment.gpu[0])
    print('model loaded to device:', config.experiment.gpu)

    data_loader = data_handler.test_dataloader()
    predictions, gts = [], []
    with torch.no_grad():
        for d in data_loader:
            x = d['x'].to(config.experiment.gpu[0])
            p = model(x)
            if config.model.n_classes > 1:
                p = torch.argmax(p, dim=1)
            else:
                p = torch.sigmoid(p).gt(0.5).long()

            predictions += p.to('cpu').tolist()
            gts += d['y'].to('cpu').tolist()
            del p
            torch.cuda.empty_cache()
    #print(predictions)
    #print(gts)
    print(f"Got {len(predictions)} predictions for {len(gts)} samples")

    report = classification_report(gts, predictions, target_names=config.model.target_labels)
    print(report)
        

if __name__ == '__main__':

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--ckpt', type=str, required=True, help="Checkpoint path to load")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Provided config not found - {args.config_path}.')

    config = OmegaConf.load(args.config)

    eval(config, args.ckpt)