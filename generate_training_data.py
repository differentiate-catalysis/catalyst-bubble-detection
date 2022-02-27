import json
import os
import random
import shutil
from typing import Tuple

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from utils import get_circle_coords, tqdm


def polar(r: float, theta: float, width: int, height: int) -> Tuple[int, int]:
    x = r * np.cos(theta) + width / 2
    y = r * np.sin(theta) + height / 2
    return int(x), int(y)


def extract_bubbles(image_root, json_root, output_dir, trim=2):
    i = 0
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    for image_name in sorted(os.listdir(image_root)):
        image = Image.open(os.path.join(image_root, image_name)).convert(mode='RGBA')
        width = image.width
        height = image.height
        image_array = np.asarray(image)
        has_label = os.path.isfile(os.path.join(json_root, image_name[:-4] + '.json'))
        if has_label:
            with open(os.path.join(json_root, image_name[:-4] + '.json')) as fp:
                json_data = json.load(fp)
                annotations = json_data['annotations']
                bubbles = []
                for circle in annotations:
                    if 'ellipse' in circle:
                        ellipse = circle['ellipse']
                        center_x = ellipse['center']['x']
                        center_y = ellipse['center']['y']
                        radius_x = ellipse['radius']['x']
                        radius_y = ellipse['radius']['y']
                        bubbles.append(np.array([center_x, center_y, radius_x, radius_y]))

            for bubble in bubbles:
                center_x = bubble[0]
                center_y = bubble[1]
                radius_x = bubble[2]
                radius_y = bubble[3]

                # Generate points for alpha channel
                rr, cc = get_circle_coords(center_y, center_x, radius_y - trim, radius_x - trim, height, width)
                y_min = np.min(rr)
                y_max = np.max(rr)
                x_min = np.min(cc)
                x_max = np.max(cc)
                bubble_image = image_array.copy()
                bubble_image[:, :, 3] = 0
                bubble_image[rr, cc, 3] = 255
                bubble_image = bubble_image[y_min: y_max + 1, x_min: x_max + 1, :]
                bubble_image = Image.fromarray(bubble_image)

                # Don't save bubbles that are cut off
                if bubble_image.height/bubble_image.width > 0.98 and bubble_image.height/bubble_image.width < 1.02:
                    bubble_image.save(os.path.join(output_dir, '%d.png' % i))
                    i += 1


def augment_image(image: Image.Image):
    image_tensor = pil_to_tensor(image)
    transform = T.Compose([
        T.ColorJitter(0.25, 0.25, 0.25),
        T.RandomApply([T.GaussianBlur(3, 0.1)]),
        T.RandomAdjustSharpness(0.2),
    ])
    image_tensor[:3, :, :] = transform(image_tensor[:3, :, :])
    image_tensor = image_tensor.permute((1, 2, 0)).numpy()
    image = Image.fromarray(image_tensor)
    return image


def generate_data(root: str, num_samples: int, min_bubble: int, max_bubble: int, width_padding: int = 200, height_padding: int = 200, bubble_trim=2):
    image_root = os.path.join(root, 'images')
    json_root = os.path.join(root, 'json')
    background_root = os.path.join(root, 'backgrounds')
    bubble_root = os.path.join(root, 'bubbles')
    output_dir = os.path.join(root, 'output')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))
    os.mkdir(os.path.join(output_dir, 'json'))
    extract_bubbles(image_root, json_root, bubble_root, trim=bubble_trim)
    num_bubbles = len(os.listdir(bubble_root))
    for sample in tqdm(range(num_samples)):
        output_json = {'annotations': []}
        background_path = os.path.join(background_root, random.choice(os.listdir(background_root)))
        background = Image.open(background_path).convert('RGBA')
        background = augment_image(background)
        num_bubbles = np.random.randint(min_bubble, max_bubble + 1)
        mask = np.zeros((background.height, background.width), dtype=np.uint8)
        for _ in range(num_bubbles):
            bubble_path = os.path.join(bubble_root, random.choice(os.listdir(bubble_root)))
            bubble_image = Image.open(bubble_path).convert('RGBA')
            bubble_image = augment_image(bubble_image)
            r = np.random.randint(min(width_padding, height_padding), min(background.width - bubble_image.width + 1 - width_padding, background.height - bubble_image.height + 1 - height_padding))/2 #* min(background.height - height_padding, background.width - width_padding)
            theta = np.random.rand() * 2 * np.pi
            x, y = polar(r, theta, background.width, background.height)
            # x = np.random.randint(width_padding, background.width - bubble_image.width + 1 - width_padding)
            # y = np.random.randint(height_padding, background.height - bubble_image.height + 1 - height_padding)
            mask_patch = mask[y: y + bubble_image.height, x: x + bubble_image.width]
            bubble_mask = np.asarray(bubble_image)[:, :, 3] // 255
            overlap_percent = np.sum(mask_patch & bubble_mask) / np.sum(bubble_mask)
            if overlap_percent > 0.55:
                continue
            bubble = {
                'ellipse': {
                    'center': {
                        'x': bubble_image.width/2 + x,
                        'y': bubble_image.height/2 + y
                    },
                    'radius': {
                        'x': bubble_image.width/2,
                        'y': bubble_image.height/2
                    }
                }
            }
            output_json['annotations'].append(bubble)
            background.alpha_composite(bubble_image, (x, y))
            mask[y: y + bubble_image.height, x: x + bubble_image.width] |= bubble_mask

        background_np = np.asarray(background)
        background_np[:, :, 3] = 255
        background = Image.fromarray(background_np)
        background.save(os.path.join(output_dir, 'images', '%d.png' % sample))
        with open(os.path.join(output_dir, 'json', '%d.json' % sample), 'w') as f:
            json.dump(output_json, f)


if __name__ == '__main__':
    generate_data('data/data_generation/pristine', 50, 5, 20, bubble_trim=-2)
