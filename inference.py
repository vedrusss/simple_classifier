import os
import sys

from sklearn.metrics import classification_report 
sys.path.append('..')

import argparse
import torch
import timm
import pytorch_lightning as pl
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from shutil import copyfile

class Classifier:
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f'Provided config not found - {config_file}.')
        self.config = OmegaConf.load(config_file)
        
        pl.seed_everything(self.config.experiment.seed)
        
        self.model = timm.create_model(
            self.config.model.arch,
            num_classes=self.config.model.n_classes,
            pretrained=False
        )
        self.gpu_id = self.config.experiment.gpu[0]
        ckp = torch.load(self.config.model.ckpt, map_location='cpu')
        state = {k.replace('student.', ''): v for k, v in ckp['state_dict'].items() if 'student' in k}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model.to(self.gpu_id)
        print('model loaded to device:', self.config.experiment.gpu)

        self.tfms = A.Compose(
            [
                A.Resize(self.config.dataset.fgvcx.crop_size, self.config.dataset.fgvcx.crop_size, cv2.INTER_LINEAR),
                A.Normalize(self.config.dataset.fgvcx.mean, self.config.dataset.fgvcx.std, 255),
                ToTensorV2()
            ]
        )
       
    def classify_image(self, img_path):
        image = np.uint8(Image.open(img_path).convert('RGB'))
        image = self.tfms(image=image)['image'].unsqueeze(0).to(self.gpu_id)
        with torch.no_grad():
            p = torch.argmax(self.model(image), dim=1)
            return p.to('cpu')[0].long().item()


def load_dataset(folder, annotation_file):
    df = pd.read_csv(annotation_file)
    samples = []
    for _, r in df.iterrows():
        img_name = r['image_name']
        label_id = r['label_id']
        label_name = r['label_name']
        path = os.path.join(folder, img_name)
        if not os.path.isfile(path): path += '.png'
        samples.append([path, label_id, label_name])
    return samples

def eval(samples, model, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    predictions, gts = [], []
    for sample in samples:
        img_path, label_id, label_name = sample
        p = model.classify_image(img_path)
        predictions.append(p)
        gts.append(label_id)
        dst_f = os.path.join(dst_folder, f"gt_{label_name}", f"pred_{model.config.model.target_labels[p]}")
        os.makedirs(dst_f, exist_ok=True)
        copyfile(img_path, os.path.join(dst_f, os.path.basename(img_path)))       

        torch.cuda.empty_cache()

    print(f"Got {len(predictions)} predictions for {len(gts)} samples")

    report = classification_report(gts, predictions, target_names=model.config.model.target_labels)
    print(report)
    print(f"Errors saved to {dst_folder}")
        

if __name__ == '__main__':

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_is_masked', type=str, default=None, help='Path to the config file of model 1')
    parser.add_argument('--config_age', type=str, default=None, help='Path to the config file of model 2')
    parser.add_argument('--errors', type=str, default=None, help="Path where to store the errors")
    parser.add_argument('--images', type=str, nargs='+', default=None, help="Images to run")
    parser.add_argument('--dataset_folder', type=str, default=None, help="Dataset root folder")
    parser.add_argument('--annotation', type=str, default=None, help="Annotation .csv file")
    args = parser.parse_args()

    model_1 = Classifier(args.config_is_masked) if args.config_is_masked else None
    model_2 = Classifier(args.config_age) if args.config_age else None

    if args.errors and args.dataset_folder and args.annotation:
        samples = load_dataset(args.dataset_folder, args. annotation)
        if model_1:
            eval(samples, model_1, args.errors)
        if model_2:
            eval(samples, model_2, args.errors)
