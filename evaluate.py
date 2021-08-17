import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from utils import Dataset, collate_fn


def evaluate(model: Module, valid_loader: DataLoader, amp: bool, gpu: int):
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        # Set model to evaluate mode (disable batchnorm and dropout, output boxes)
        model.eval()
        total_iou = 0
        total_loss = 0
        num_samples = valid_loader.batch_size * len(valid_loader)
        for images, targets in tqdm(valid_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                for i in range(len(targets)):
                    output = outputs[i]
                    target = targets[i]
                    iou = torchvision.ops.box_iou(output['boxes'], target['boxes'])
                    dims = iou.shape
                    matrix = torch.zeros((max(dims[0], dims[1]), max(dims[0], dims[1])))
                    matrix[:dims[0], :dims[1]] = iou
                    total_iou += torch.sum(torch.diagonal(matrix))
                # Must set model to train mode to get loss
                model.train()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values()).item()
                model.eval()
            total_loss += loss * valid_loader.batch_size
        print('--- evaluation result ---')
        print('loss: %.5f, iou %.5f' % (total_loss / num_samples, total_iou / num_samples))

    return total_loss / num_samples, total_iou / num_samples

def apply(args: SimpleNamespace):
    saved_model = torch.load('saved/best.pth')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(saved_model['model_state_dict'])

    if args.gpu != -1:
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cpu')
    model.to(device)

    with torch.no_grad():
        model.eval()
        for filename in tqdm(os.listdir(args.dir)):
            image = Image.open(os.path.join(args.dir, filename))
            image = image.convert(mode='RGB')
            image = T.ToTensor()(image)
            # This is to handle evaluation with low precision
            with torch.cuda.amp.autocast(enabled=args.amp):
                bounding_boxes = model([image.to(device)])
                boxes = bounding_boxes[0]['boxes']
                # Write bounding boxes to a file and save images cropped by bounding boxes
                with open(os.path.join(args.dir, args.save), 'a') as f:
                    for i in range(boxes.shape[0]):
                        f.write(filename)
                        f.write(' ')
                        f.write(str(boxes[i])) #TODO: fix this
                        f.write('\n')

                        cropped_image = image[:, int(boxes[i, 1].cpu().item()):int(boxes[i,3].cpu().item()), int(boxes[i, 0].cpu().item()):int(boxes[i, 2].cpu().item())]
                        cropped_image = cropped_image.permute((1, 2, 0))
                        cropped_image = Image.fromarray(np.uint8((cropped_image.numpy())*255), 'RGB')
                        if not os.path.isdir(os.path.join(args.dir, 'crops')):
                            os.mkdir(os.path.join(args.dir, 'crops'))
                        if not os.path.isdir(os.path.join(args.dir, 'crops/', filename[:-4])):
                            os.mkdir(os.path.join(args.dir, 'crops/', filename[:-4]))
                        cropped_image.save(os.path.join(args.dir, 'crops/', filename[:-4], str(i) + '.jpg'))

    print('finished writing file')

def test(args: SimpleNamespace):
    # Load previous model
    saved_model = torch.load('./saved/best.pth')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(saved_model['model_state_dict'])

    data = Dataset(os.path.join(args.root, args.dir), [], False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=True)

    if args.gpu != -1:
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cpu')
    model.to(device)

    loss_test, iou_test = evaluate(model, data_loader, args.amp, args.gpu)
