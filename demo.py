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
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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
       
    def classify_image(self, image):
        image = self.tfms(image=image)['image'].unsqueeze(0).to(self.gpu_id)
        with torch.no_grad():
            p = torch.argmax(self.model(image), dim=1)
            return p.to('cpu')[0].long().item()


class DataReader:
    def __init__(self, filename):
        self.single_image = False
        if filename == '': self.src = cv2.VideoCapture(0)
        ext = filename.split('.')[-1]
        if ext.upper() in ('AVI', 'MP4'): self.src = cv2.VideoCapture(filename)
        else: 
            self.src = cv2.imread(filename)
            self.single_image = True
    
    def next(self):
        if self.single_image:
            res = self.src
            self.src = None
            return res
        if self.src.isOpened():
            ret, frame = self.src.read()
            if not ret is None: return frame
        return None  


def main(face_detector, model_1, model_2, src):
    frame = src.next()
    i = 1
    while not frame is None:
        dets = face_detector.detectMultiScale(frame, 1.1, 4)
        print(f"frame {i} - {len(dets)} faces detected:")
        for (x,y,w,h) in dets:
            crop = frame[y:y+h, x:x+w, :]
            if model_1:
                prediction_1 = model_1.config.model.target_labels[model_1.classify_image(crop)]
                if prediction_1 == "with_mask":
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    prediction_2 = ""
                else:
                    prediction_2 = model_2.config.model.target_labels[model_2.classify_image(crop)] if model_2 else None
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                print(f"\t{prediction_1} {prediction_2}")
            
        cv2.imshow('win', frame)
        if len(dets): key = cv2.waitKey()
        else: key = cv2.waitKey(33)
        if key == 27:
            break
        i += 1
        frame = src.next()

        torch.cuda.empty_cache()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_is_masked', type=str, default=None, help='Path to the config file of model 1')
    parser.add_argument('--config_age', type=str, default=None, help='Path to the config file of model 2')
    parser.add_argument('--input', type=str, default='', help="Provide video file, image or keep empty to use web cam")
    args = parser.parse_args()

    face_detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    model_1 = Classifier(args.config_is_masked) if args.config_is_masked else None
    model_2 = Classifier(args.config_age) if args.config_age else None

    main(face_detector, model_1, model_2, DataReader(args.input))
