from functools import partial
import torch
from torch import optim
import torch.nn.functional as F
import losses
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import torch_optimizer



def get_scheduler(scheduler_config, optimizer):
    
    name = scheduler_config.name
    monitor = scheduler_config.monitor

    if name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=scheduler_config.mode,
                                                               patience=scheduler_config.patience,
                                                               factor=scheduler_config.factor,
                                                               min_lr=scheduler_config.min_lr)
        return dict(scheduler=scheduler, monitor=monitor)

    elif name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=0.000001)
        return dict(scheduler=scheduler, monitor=monitor)

    elif name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config.gamma)
        return dict(scheduler=scheduler, monitor=monitor)

    else:
        raise ValueError(f'{name} not in schedulers')


class BinaryEngine(pl.LightningModule):
    def __init__(self, model, criterion=None, opt=None, lr=None, wd=None, scheduler_conf=None, T=3, alpha=1, th=0.5):
        super().__init__()

        self.model = model
        self.T = T
        self.alpha = alpha
        self.criterion = partial(getattr(losses, criterion), temperature=T, alpha=alpha) # getattr(F, criterion) if criterion else None
        self.opt = opt 
        self.lr = lr
        self.wd = wd
        self.scheduler_conf = scheduler_conf
        self.th = th


    def forward(self, x):
        
        logit = self.model(x)

        return logit


    def training_step(self, batch, batch_index):
        
        # keep BN stats untouchable
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

        x, y = batch['x'], batch['y']
        
        logit = self(x)
        bce_loss = self.criterion(logit, y.view(logit.size()))

        self.log(f"train_bce_loss", bce_loss)

        return bce_loss
    

    def validation_step(self, batch, batch_index):

        x, y = batch['x'], batch['y']

        logit = self(x)

        val_bce_loss = self.criterion(logit, y.view(logit.size()))
        acc = accuracy(torch.sigmoid(logit).gt(self.th).long(), y.view(logit.size()).long())

        self.log("val_bce_loss", val_bce_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return val_bce_loss


    def test_step(self, batch, batch_index):

        x, y = batch['x'], batch['y']

        logit = self(x)
        acc = accuracy(torch.sigmoid(logit).gt(self.th).long(), y.view(logit.size()).long())

        self.log("test_acc", acc, prog_bar=True)
        
        return acc
        

    def configure_optimizers(self):

        optimizer = getattr(optim, self.opt)(self.model.classifier.parameters(), lr=self.lr, weight_decay=self.wd)

        scheduler_config = self.scheduler_conf
        
        if scheduler_config is not None:
            scheduler = get_scheduler(scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

class MultiEngine(pl.LightningModule):
    def __init__(self, student=None, criterion=None, opt=None, lr=None, wd=None, scheduler_conf=None):
        super().__init__()

        self.student = student
        self.criterion = partial(getattr(losses, criterion)) if criterion else criterion
        self.opt = opt 
        self.lr = lr
        self.wd = wd
        self.scheduler_conf = scheduler_conf
        self.genus_head = hasattr(self.student, 'genus_head')

        print('HEAD',hasattr(self.student, 'genus_head'))

    def forward(self, x):
        
        if self.genus_head:

            x = self.student.forward_features(x)
            x = self.student.global_pool(x)

            if self.student.drop_rate:
                x = F.dropout(x, p=self.student.drop_rate, training=self.student.training)

            return {'y': self.student.classifier(x), 'g': self.student.genus_head(x)}

        return  {'y': self.student(x)}

    def training_step(self, batch, batch_index):
        
        x = batch['x']
        logits = self(x)

        total_loss = 0.0
        for logit_name, logit_val in logits.items():

            gt = batch[logit_name]
            losses = self.criterion(logit_val, gt)
        
            for lossname, lossval in losses.items():
                self.log(f"train_{logit_name}_{lossname}", lossval)

            total_loss += losses['overall']

        self.log(f"train_total_loss", total_loss)

        return total_loss
    
    def validation_step(self, batch, batch_index):

        x = batch['x'] 
        logits = self(x)

        for logit_name, logit_val in logits.items():
            
            gt = batch[logit_name]
            loss = F.cross_entropy(logit_val, gt)

            preds = torch.argmax(logit_val, dim=1)
            acc = accuracy(preds, gt)

            self.log(f"val_{logit_name}_loss", loss, prog_bar=True)
            self.log(f"val_{logit_name}_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_index):

        x, y = batch['x'], batch['y']

        logits = self(x)
        
        preds = torch.argmax(logits['y'], dim=1)
        acc = accuracy(preds, y)

        self.log("test_acc", acc, prog_bar=True)

        return acc

    def configure_optimizers(self):
        
        if self.opt in ['SGD', 'Adam', 'AdamW', 'RMSprop']:
            optimizer = getattr(optim, self.opt)(filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.lr, weight_decay=self.wd)

        # elif self.opt == 'AdaBelief':
        #     optimizer = AdaBelief(filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.lr, weight_decay=self.wd, weight_decouple=True, rectify=False)
        else:
            optimizer = getattr(torch_optimizer, self.opt)(filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.lr, weight_decay=self.wd)

        scheduler_config = self.scheduler_conf

        print(scheduler_config)
        
        if scheduler_config is not None:
            scheduler = get_scheduler(scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer