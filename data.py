import os
import json
from functools import partial

import pandas as pd
import numpy as np 
import torch
import cv2
import albumentations as A
from PIL import Image

from albumentations.pytorch.transforms import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class CenterCropFraction(A.CenterCrop):
    def __init__(self, fraction=0.8, always_apply=False, p=1.0):
        super().__init__(0, 0, always_apply, p)
        self.fraction = fraction

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        self.height = int(min(h, w) * self.fraction)
        self.width = self.height
        return {}

    def get_transform_init_args_names(self):
        return "fraction",


class RandomCropFraction(A.RandomCrop):
    def __init__(self, fraction=0.8, always_apply=False, p=1.0):
        super().__init__(0, 0, always_apply, p)
        self.fraction = fraction

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        self.height = int(min(h, w) * self.fraction)
        self.width = self.height
        return {}

    def get_transform_init_args_names(self):
        return "fraction",
    
    
def load_fgvcx(jspath = '/data/fgvcx_kaggle_danish/train_val_anno/train.json'):

    with open(jspath, 'r') as f:
        train_anno = json.load(f)

    fgvcx_anno = pd.DataFrame.from_dict(train_anno['annotations'])

    categ_map = {}
    
    for d in train_anno['categories']:
        categ_map[str(d['id'])] = {'name': d['name'].lower(), 'supercategory': d['supercategory'].lower()}

    fgvcx_anno['label'] = [categ_map[str(n)]['name'].lower() for n in fgvcx_anno.category_id]
    fgvcx_anno['supercategory'] = [categ_map[str(n)]['supercategory'].lower() for n in fgvcx_anno.category_id]
    
    fgvcx_anno['label_words'] = [len(v.split(' ')) for  v in fgvcx_anno.label]
    
    img_path_dict = {v['id']: v['file_name'].replace('images/', '') for v in train_anno['images']}
    
    fgvcx_anno['image_path'] = fgvcx_anno.image_id.map(img_path_dict)
    
    return (fgvcx_anno, categ_map)


def open_img(img_path):
    image = Image.open(img_path).convert('RGB')
    return np.uint8(image)


def proc_image(img_path, side_ratio=0.8, size=299):
    
    img = open_img(img_path)
    
    crop_size = int(min(img.shape[:2]) * side_ratio)
    
    img = A.CenterCrop(crop_size, crop_size, 1)(image=img)['image']
    img = A.Resize(size, size, cv2.INTER_AREA)(image=img)['image']
    
    img = torch.from_numpy(img / 255 * 2 - 1).float()
    
    return img.permute(2, 0, 1).unsqueeze(0)

    
class FGVCX(Dataset):
    def __init__(self, data_list, root, tfms, train):

        self.data_list = data_list
        self.root = root
        self.tfms = tfms
        self.train = train
        
    def __getitem__(self, x):
        datum = self.data_list.loc[x]
        path = f'{self.root}/{datum.image_name}'
        if not os.path.isfile(path): path = path + '.png'

        try: image = open_img(path)
        except:
            print(x, datum.image_name, datum.label_id, datum.label_name, path)
            quit()

        if self.tfms:
            image = self.tfms(image=image)['image']

        y = torch.tensor(datum.label_id).long()
        
        return dict(x=image, y=y)
    
    def __len__(self):
        return self.data_list.shape[0]


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.class_map = None

        self.train_transform = A.Compose(
            [
                ##RandomCropFraction(fraction=self.config.crop_fraction), 
                A.Resize(self.config.crop_size, self.config.crop_size, cv2.INTER_LINEAR),
                A.HorizontalFlip(),
                ##A.VerticalFlip(),
                A.Normalize(self.config.mean, self.config.std, 255),
                ToTensorV2()
            ]
        )

        self.valid_transform = A.Compose(
            [
                ##CenterCropFraction(fraction=self.config.crop_fraction), 
                A.Resize(self.config.crop_size, self.config.crop_size, cv2.INTER_LINEAR),
                A.Normalize(self.config.mean, self.config.std, 255),
                ToTensorV2()
            ]
        )
        

    def setup(self, stage=None):

        self.train_dataset = FGVCX(
            pd.read_csv(self.config.train_list),
            root=self.config.root,
            tfms=self.train_transform,
            train=True
        )

        self.val_dataset = FGVCX(
            pd.read_csv(self.config.val_list),
            root=self.config.root,
            tfms=self.valid_transform,
            train=False
        )

        self.test_dataset = FGVCX(
            pd.read_csv(self.config.test_list),
            root=self.config.root,
            tfms=self.valid_transform,
            train=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)
