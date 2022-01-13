import torch
import argparse
from models.detector.retinanet import RetinaNet
from module.detector import Detector
from utils.module_select import (get_cls_subnet, get_fpn, get_model,
                                 get_reg_subnet)
from utils.yaml_helper import get_train_configs


 
 
def main(cfg): 
    backbone = get_model(cfg['backbone'])
    fpn = get_fpn(cfg['fpn'])
    cls_sub = get_cls_subnet(cfg['cls_subnet'])
    reg_sub = get_reg_subnet(cfg['reg_subnet'])
    model = RetinaNet(backbone, fpn, cls_sub, reg_sub, cfg['classes'], cfg['in_channels'])
    if torch.cuda.is_available:
        model = model.to('cuda')
 
    model_module = Detector.load_from_checkpoint( 
        '/home/insig/Detection_RetinaNet/saved/ResNet_RetinaNet_Test/version_0/checkpoints/last.ckpt', model=model 
    ) 
    model_module.eval() 
 
    # Convert PyTorch To ONNX 
    dumTensor = torch.rand(1, cfg['in_channels'], cfg['input_size'], cfg['input_size'])
    if torch.cuda.is_available:
        dumTensor = dumTensor.to('cuda') 
    torch.onnx.export(model_module.model, dumTensor, cfg['model']+'.onnx', 
                      export_params=True, opset_version=9, do_constant_folding=True,
                      input_names=['input'], output_names=['classifications', 'regressions'])
 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--cfg', required=True, type=str, help='Train config file') 
    args = parser.parse_args() 
    cfg = get_train_configs(args.cfg) 
    main(cfg)
