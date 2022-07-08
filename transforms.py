import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from kornia.enhance import adjust_log, equalize_clahe
from kornia.color import rgb_to_hsv, hsv_to_rgb
from skimage import exposure
from torch.nn import Module


class Compose(Module):
    def __init__(self, transforms: List[Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(Module):
    def __init__(self, gpu: int = -1):
        if gpu != -1:
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.to_tensor(image).to(device=self.device)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
                if 'masks' in target:
                    target['masks'] = target['masks'].flip(-1)
        return image, target


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height = F._get_image_size(image)
                target['boxes'][:, [1, 3]] = height - target['boxes'][:, [3, 1]]
                if 'masks' in target:
                    target['masks'] = target['masks'].flip(-2)
        return image, target


def transform_boxes(image: torch.Tensor, target: Dict[str, torch.Tensor], transform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Method for applying matrix transformations to bounding boxes.
    Parameters
    ----------
    image: torch.Tensor
        Tensor containing the image data, of shape (3, H, W)
    target: Dict[str, torch.Tensor]
        The dictionary containing the classes, bounding boxes, areas, etc.
    transform: torch.Tensor
        Matrix used to transform bounding boxes.
    Returns
    -------
    areas, out_boxes
        Tensor of transformed areas and bounding boxes
    '''
    # Add all 4 corners
    boxes = torch.empty((target['boxes'].shape[0], 8))
    boxes[:, :4] = target['boxes'].clone().detach()

    # Bottom left
    boxes[:, 4] = boxes[:, 0]
    boxes[:, 5] = boxes[:, 3]

    # Top right
    boxes[:, 6] = boxes[:, 2]
    boxes[:, 7] = boxes[:, 1]

    # Center the boxes
    width, height = F._get_image_size(image)
    boxes[:, [0, 2, 4, 6]] -= width / 2
    boxes[:, [1, 3, 5, 7]] -= height / 2

    # Apply matrix transformation
    boxes[:, [0, 1]] = (transform @ boxes[:, [0, 1]].permute((1, 0))).permute((1, 0))
    boxes[:, [2, 3]] = (transform @ boxes[:, [2, 3]].permute((1, 0))).permute((1, 0))
    boxes[:, [4, 5]] = (transform @ boxes[:, [4, 5]].permute((1, 0))).permute((1, 0))
    boxes[:, [6, 7]] = (transform @ boxes[:, [6, 7]].permute((1, 0))).permute((1, 0))

    # Un-center the boxes
    boxes[:, [0, 2, 4, 6]] += width / 2
    boxes[:, [1, 3, 5, 7]] += height / 2

    # Choose the right x and y combos for the boxes
    out_boxes = torch.empty_like(target['boxes'])
    out_boxes[:, 0] = torch.maximum(torch.amin(boxes[:, [0, 2, 4, 6]], dim=-1), torch.zeros(1))
    out_boxes[:, 1] = torch.maximum(torch.amin(boxes[:, [1, 3, 5, 7]], dim=-1), torch.zeros(1))
    out_boxes[:, 2] = torch.minimum(torch.amax(boxes[:, [0, 2, 4, 6]], dim=-1), torch.tensor([width - 1]))
    out_boxes[:, 3] = torch.minimum(torch.amax(boxes[:, [1, 3, 5, 7]], dim=-1), torch.tensor([height - 1]))

    # Remove degenerate boxes:
    nonnegative_vals_tensor = torch.sum(out_boxes >= 0, dim=-1) == 4
    nonnegative_width_tensor = (out_boxes[:, 2] - out_boxes[:, 0]) > 0
    nonnegative_height_tensor = (out_boxes[:, 3] - out_boxes[:, 1]) > 0
    total_tensor = nonnegative_height_tensor * nonnegative_vals_tensor * nonnegative_width_tensor
    proper_instances = torch.where(total_tensor == 1)
    out_boxes = out_boxes[proper_instances]
    areas = (out_boxes[:, 2] - out_boxes[:, 0]) * (out_boxes[:, 3] - out_boxes[:, 1])
    return areas, out_boxes


def target_from_masks(masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    '''
    Given masks, convert this to a full target dictionary (generate areas, bbox, etc.)
    Parameters
    ----------
    masks: torch.Tensor
        Tensor of instance segmentations. Shape (N, H, W)
    Returns
    -------
    target
        Updated dictionary with masks, areas, bboxes, etc.
    '''
    target = {}

    # We can instances with no mask by summing and checking for a positive number
    nonzero_instances = torch.where(torch.sum(masks, dim=[-1, -2]) > 0)[0]
    if nonzero_instances.shape[0] > 0:
        masks = masks[nonzero_instances]

        # Now we need to check for masks where the max and min are the same
        boxes = torch.empty((masks.shape[0], 4), dtype=torch.float32)
        areas = torch.empty(masks.shape[0])
        proper_instances = list(range(masks.shape[0]))
        for i in range(masks.shape[0]):
            rows, cols = torch.where(masks[i] == 1)
            x_0 = torch.amin(cols)
            y_0 = torch.amin(rows)
            x_1 = torch.amax(cols)
            y_1 = torch.amax(rows)
            boxes[i, :] = torch.tensor([x_0, y_0, x_1, y_1], dtype=torch.float32)
            areas[i] = (x_1 - x_0) * (y_1 - y_0)

            # If an instance has width or height 0, discard it
            if x_0 == x_1 or y_0 == y_1:
               proper_instances.remove(i)
        masks = masks[proper_instances]
        boxes = boxes[proper_instances]
        areas = areas[proper_instances]
        target['labels'] = torch.ones(boxes.shape[0], dtype=torch.int64)
        target['iscrowd'] = torch.zeros(boxes.shape[0], dtype=torch.uint8)
    else:
        masks = torch.tensor([[[]]], dtype=torch.uint8)
        boxes = torch.tensor([[]], dtype=torch.float32)
        areas = torch.tensor([], dtype=torch.float32)
        target['labels'] = torch.tensor([], dtype=torch.int64)
        target['iscrowd'] = torch.tensor([], dtype=torch.uint8)
    target['masks'] = masks
    target['boxes'] = boxes
    target['areas'] = areas
    return target


class RandomRotation(T.RandomRotation):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        degrees = self.get_params(self.degrees)
        rad = degrees * math.pi / 180
        image = F.rotate(image, degrees)
        if target is not None:
            if 'masks' in target:
                mask = F.rotate(target['masks'], degrees)
                corrected_target = target_from_masks(mask)

                # If we only have degenerate boxes, ignore the rotation
                if 0 in corrected_target['boxes'].shape:
                    return image, target
                for key, value in corrected_target.items():
                    target[key] = value
            else:
                matrix = torch.tensor([[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]], dtype=torch.float32)
                target['areas'], target['boxes'] = transform_boxes(image, target, matrix)
        return image, target


class RandomApply(T.RandomApply):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if self.p < torch.rand(1):
            return image, target
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter(T.ColorJitter):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = super().forward(image)
        return image, target


class RandomAdjustSharpness(T.RandomAdjustSharpness):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = super().forward(image)
        return image, target


class GaussianBlur(T.GaussianBlur):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = super().forward(image)
        return image, target


class RandomSaltAndPepper(Module):
    def __init__(self, p: float = 0.01):
        super().__init__()
        self.p = p
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        probs = torch.randn(image.shape[1:])
        image[:, (probs < self.p / 2)] = 255
        image[:, (probs > 1 - self.p / 2)] = 0
        return image, target


class Normalize(T.Normalize):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = super().forward(image)
        return image, target


class LogGamma(Module):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = adjust_log(image)  # Runs Log Adjustment through kornia.
        return image, target


class CLAHE(Module):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = image / 255
        image = image.permute(2, 0, 1)
        image = rgb_to_hsv(image, eps=1e-08)
        image[2] = equalize_clahe(image[2], clip_limit=1.5, grid_size=(4,4))
        image = hsv_to_rgb(image)
        return image.permute(1, 2, 0)


class ContrastStretchInt(Module):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Compute the 2nd and 98th percentiles
        q = torch.Tensor([0.02, 0.98])
        p2, p98 = torch.quantile(image, q)

        # Clip the image to these percentiles
        image = image.clip(p2, p98)

        # Stretch the image back to 0..255
        image -= p2
        image /= (p98 - p2)
        image *= 255
        return image.int(), target

class ContrastStretchFloat(Module):
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Compute the 2nd and 98th percentiles
        q = torch.Tensor([0.02, 0.98])
        p2, p98 = torch.quantile(image, q)

        # Clip the image to these percentiles
        image = image.clip(p2, p98)

        # Stretch the image back to 0..1
        image -= p2
        image /= (p98 - p2)
        return image, target


transform_mappings = {
    'log_gamma': LogGamma(),
    'clahe': CLAHE(),
    'contrast_stretch_float': ContrastStretchFloat(),
    'contrast_stretch_int': ContrastStretchInt(),
    'horizontal_flip': RandomHorizontalFlip(0.5),
    'vertical_flip': RandomVerticalFlip(0.5),
    'rotation': RandomRotation(80),
    'gray_balance_adjust': RandomApply([ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p=0.5),
    'blur': RandomApply([GaussianBlur(3)], p=0.5),
    'sharpness': RandomAdjustSharpness(1.2, 0.5),
    'saltandpepper': RandomSaltAndPepper(p=0.01)
}
