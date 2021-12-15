import json
import os
from types import SimpleNamespace
from typing import List

from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn

from utils import compute_mean_and_std


def faster_rcnn(image_mean: List[float], image_std: List[float]) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def mask_rcnn(image_mean: List[float], image_std: List[float]) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    return model


def retina_net(image_mean: List[float], image_std: List[float]) -> RetinaNet:
    model = retinanet_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head = RetinaNetHead(in_features, num_anchors, 2)
    return model


def model_mappings(args: SimpleNamespace):
    if not args.imagenet_stats:
        if args.stats_file:
            stats = args.stats_file
        else:
            stats = os.path.join(args.root, args.name, 'stats.json')
            if not os.path.isfile(os.path.join(args.root, 'stats.json')):
                compute_mean_and_std([os.path.join(args.root, args.name, 'train'), os.path.join(args.root, args.name, 'validation')], args.image_size)
        with open(stats) as f:
            stats = json.load(f)
            mean = stats['mean']
            std = stats['std']
    else:
        mean = None
        std = None
    model_map = {
        'faster_rcnn': faster_rcnn(mean, std),
        'mask_rcnn': mask_rcnn(mean, std),
        'retina_net': retina_net(mean, std),
    }
    return model_map[args.model]


model_keys = set(['faster_rcnn', 'mask_rcnn', 'retina_net'])
