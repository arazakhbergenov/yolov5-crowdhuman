"""
Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ python export_onnx.py --weights crowdhuman_yolov5m.pt --img 640 --batch 1
"""

import argparse

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging
from utils.torch_utils import select_device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=device)
    if half:
        model.half()  # to FP16

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size)).type_as(next(model.parameters()))  # image size(1,3,320,192) iDetection

    # Update model
    # for k, m in model.named_modules():
    #     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    #     if isinstance(m, models.common.Conv):  # assign export-friendly activations
    #         if isinstance(m.act, nn.Hardswish):
    #             m.act = Hardswish()
    #         elif isinstance(m.act, nn.SiLU):
    #             m.act = SiLU()
    #     # if isinstance(m, models.yolo.Detect):
    #     #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
