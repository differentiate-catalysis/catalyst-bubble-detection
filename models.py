import json
import os
from types import SimpleNamespace
from typing import List, Optional

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn
from torchvision.models.resnet import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from utils import compute_mean_and_std, compute_mean_and_std_video



def resnet_fpn_backbone(
    backbone,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def faster_rcnn(image_mean: Optional[List[float]], image_std: Optional[List[float]]) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def mask_rcnn(image_mean: Optional[List[float]], image_std: Optional[List[float]]) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    return model


def retina_net(image_mean: Optional[List[float]], image_std: Optional[List[float]]) -> RetinaNet:
    model = retinanet_resnet50_fpn(pretrained=True, image_mean=image_mean, image_std=image_std)
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head = RetinaNetHead(in_features, num_anchors, 2)
    return model


def simclr_faster_rcnn(image_mean: Optional[List[float]], image_std: Optional[List[float]], checkpoint_path: str) -> FasterRCNN:
    if checkpoint_path is None:
        return None
    resnet_backbone = resnet50(pretrained=True)
    with open(checkpoint_path, 'rb') as f:
        state_dict = torch.load(f)
        resnet_backbone.load_state_dict(state_dict)
    trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    backbone = resnet_fpn_backbone(resnet_backbone, trainable_backbone_layers)
    model = faster_rcnn(image_mean, image_std)
    model.backbone = backbone
    return model


def model_mappings(args: SimpleNamespace):
    if not args.imagenet_stats:
        if args.stats_file:
            stats = args.stats_file
        else:
            if args.video:
                compute_mean_and_std_video(os.path.join(args.root, args.name), args.video, (1920, 1080))
            stats = os.path.join(args.root, args.name, 'stats.json')
            if not os.path.isfile(os.path.join(args.root, args.name, 'stats.json')):
                print('Stats file not found! Recomputing...')
                compute_mean_and_std([os.path.join(args.root, args.name, 'train'), os.path.join(args.root, args.name, 'validation')], args.image_size)
        with open(stats) as f:
            stats = json.load(f)
            mean = stats['mean']
            std = stats['std']
    else:
        mean = None
        std = None
    model_map = {
        'simclr_faster_rcnn': simclr_faster_rcnn(mean, std, args.simclr_checkpoint),
        'faster_rcnn': faster_rcnn(mean, std),
        'mask_rcnn': mask_rcnn(mean, std),
        'retina_net': retina_net(mean, std),
    }
    return model_map[args.model]


model_keys = set(['faster_rcnn', 'mask_rcnn', 'retina_net', 'simclr_faster_rcnn'])


if __name__ == '__main__':
    model = simclr_faster_rcnn(None, None, 'saved/resnet50_checkpoint_100.tar')
    print(model)
