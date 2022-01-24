import argparse
import os
import random

import cv2
import numpy as np
import torch

from models.detector.retinanet import RetinaNet
from module.detector import Detector
from module.transformer import Transformer
from utils.module_select import (get_cls_subnet, get_fpn, get_model,
                                 get_reg_subnet)
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs

from torchsummary import summary

def parse_names(names_file):
    names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def visualize_detection(image, box, class_name, conf, color):
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

    caption = f'{class_name} {conf:.2f}'
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return image


def bin_load(bin_path):
    num_cls_pred = np.fromfile("./inference/dump/classification_out.bin", dtype='float32')
    num_reg_pred = np.fromfile("./inference/dump/regression_out.bin", dtype='float32')
    cls_pred = torch.Tensor(num_cls_pred)
    reg_pred = torch.Tensor(num_reg_pred)
    if torch.cuda.is_available:
        cls_pred = cls_pred.to('cuda')
        reg_pred = reg_pred.to('cuda')
    cls_pred = torch.reshape(cls_pred, (1, 20, -1))
    reg_pred = torch.reshape(reg_pred, (1, 4, -1))
    return cls_pred, reg_pred
    

def main(cfg, image_name, save):
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (320, 320))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp = image_inp.cuda()

    # Load trained model
    backbone = get_model(cfg['backbone'])
    fpn = get_fpn(cfg['fpn'])
    cls_sub = get_cls_subnet(cfg['cls_subnet'])
    reg_sub = get_reg_subnet(cfg['reg_subnet'])
    model = RetinaNet(backbone, fpn, cls_sub, reg_sub,
                      cfg['classes'], cfg['in_channels'])
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = Detector.load_from_checkpoint(
        '/home/insig/Detection_RetinaNet/saved/ResNet_RetinaNet_Test/version_0/checkpoints/last.ckpt',
        model=model)
    model_module.eval() 
    # summary(model_module, input_size=(cfg['in_channels'], cfg['input_size'], cfg['input_size']))

    transformer = Transformer()

    # inference
    cls_pred, reg_pred = model_module(image_inp)
    confidences, cls_idxes, boxes = transformer(
        [image_inp, cls_pred, reg_pred])

    confidences = confidences[0]
    cls_idxes = cls_idxes[0]
    boxes = boxes[0]
    idxs = np.where(confidences.cpu() > 0.3)

    for i in range(idxs[0].shape[0]):
        box = boxes[idxs[0][i]]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        name = names[int(cls_idxes[idxs[0][i]])]
        conf = confidences[idxs[0][i]]
        color = colors[int(cls_idxes[idxs[0][i]])]
        image = visualize_detection(image, (x1, y1, x2, y2), name, conf, color)

    # cv2.imshow('test', image)
    # cv2.waitKey(0)
    cv2.imwrite('./inference/result/inference.png', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--save', action='store_true',
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    main(cfg, './inference/sample/(1).jpg', args.save)
