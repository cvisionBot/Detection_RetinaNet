import argparse

from utils.utility import make_model_name

import albumentations
import albumentations.pytorch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining

from dataset.detection import yolo_format
from utils.module_select import get_cls_subnet, get_fpn, get_model, get_reg_subnet
from utils.yaml_helper import get_train_configs

from module.detector import Detector
from models.detector.retinanet import RetinaNet
import platform

def add_experimental_callbacks(cfg, train_callbacks):
    options = {
        'SWA': StochasticWeightAveraging(),
        'QAT': QuantizationAwareTraining()
    }
    callbacks = cfg['experimental_options']['callbacks']
    if callbacks:
        for option in callbacks:
            train_callbacks.append(options[option])

    return train_callbacks


def train(cfg, ckpt=None):
    input_size = cfg['input_size']
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(input_size, input_size, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    valid_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    data_module = yolo_format.YoloFormat(
        train_list=cfg['train_list'], val_list=cfg['val_list'],
        workers=cfg['workers'], batch_size=cfg['batch_size'],
        train_transforms=train_transforms, val_transforms=valid_transform
    )

    backbone = get_model(cfg['backbone'])
    fpn = get_fpn(cfg['fpn'])
    cls_sub = get_cls_subnet(cfg['cls_subnet'])
    reg_sub = get_reg_subnet(cfg['reg_subnet'])

    model = RetinaNet(backbone, fpn, cls_sub, reg_sub,
                      cfg['classes'], cfg['in_channels'])
    model_module = Detector(
        model, cfg, epoch_length=data_module.train_dataloader().__len__())
    
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor='val_loss', save_last=True,
                        every_n_epochs=cfg['save_freq']),
    ]

    callbacks = add_experimental_callbacks(cfg, callbacks)

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'],
                                 make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp' if platform.system() != 'Windows' else None,
        plugins=DDPPlugin() if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        resume_from_checkpoint=ckpt,
        **cfg['trainer_options'])
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--ckpt', required=False, type=str,
                        help='Train checkpoint')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    train(cfg, args.ckpt)
