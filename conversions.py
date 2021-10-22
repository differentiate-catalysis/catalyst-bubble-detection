import json
import os
import skimage.draw
import numpy as np
import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

from PIL import Image, ImageDraw
from tqdm import tqdm

def v7labstobasicjson(infile: str, outfile: str=None) -> dict:
    with open(infile, 'r') as infp:
        indata = json.load(infp)
        outdir = {}
        outdir['width'] = indata['image']['width']
        outdir['height'] = indata['image']['height']
        outdir['filename'] = indata['image']['filename']
        outdir['bubbles'] = []
        for annotate in indata['annotations']:
            bubble = annotate['ellipse']
            bubble_data = {
                "center": [bubble["center"]["x"], bubble["center"]["y"]],
                "radius": bubble["radius"]["x"]
            }
            outdir['bubbles'].append(bubble_data)
    if outfile:
        with open(outfile, 'w') as outfp:
            json.dump(outdir, outfp, indent=2)
    return outdir

def getjsontype(json_dict: dict) -> bool:
    #Returns True if the json_dict is in basic form, False if not (probably in v7 labs form and should convert)
    return not ({'bubbles', 'height', 'width', 'filename'} - set(json_dict.keys()))

def gen_label_images(indir: str, outdir: str) -> None:
    in_files = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith('.json')]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print("Generating labels")
    for file in tqdm(in_files):
        with open(file, 'r') as fp:
            json_data = json.load(fp)
        if not getjsontype(json_data):
            json_data = v7labstobasicjson(file)
        image = Image.new('1', (json_data['width'], json_data['height']))
        drawer = ImageDraw.Draw(image)
        for bubble in json_data['bubbles']:
            drawer.ellipse([bubble['center'][0] - bubble['radius'], bubble['center'][1] - bubble['radius'], bubble['center'][0] + bubble['radius'], bubble['center'][1] + bubble['radius']], fill = 1)
        image.save(os.path.join(outdir, os.path.splitext(os.path.split(json_data['filename'])[1])[0] + '.png'))

@python_app
def convert_to_targets(image_root, json_root, output_dir, image_name):
    image = Image.open(os.path.join(image_root, image_name)).convert(mode='RGB')
    width = image.width
    height = image.height
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

    # Allocate space for mask, boxes, and area
    mask = np.zeros((len(bubbles), height, width), dtype=np.uint8)
    boxes = np.empty((len(bubbles), 4), dtype=np.float32)
    areas = np.empty(len(bubbles), dtype=np.float32)
    for i, bubble in enumerate(bubbles):
        center_x = bubble[0]
        center_y = bubble[1]
        radius_x = bubble[2]
        radius_y = bubble[3]
        # Get coordinates for bounding box, making sure to bound by image size
        x_0 = max(center_x - radius_x, 0)
        y_0 = max(center_y - radius_y, 0)
        x_1 = min(center_x + radius_x, width - 1)
        y_1 = min(center_y + radius_y, height - 1)
        # Save box and area
        boxes[i, :] = np.array([x_0, y_0, x_1, y_1], dtype=np.float32)
        areas[i] = (x_1 - x_0) * (y_1 - y_0)
        # Generate points for mask
        rr, cc = skimage.draw.ellipse(center_y, center_x, radius_y, radius_x)
        # Limit these to only points in the image
        cc = cc[rr < height]
        rr = rr[rr < height]
        rr = rr[cc < width]
        cc = cc[cc < width]
        cc = cc[rr >= 0]
        rr = rr[rr >= 0]
        rr = rr[cc >= 0]
        cc = cc[cc >= 0]
        mask[i, rr, cc] = 1

    np.save(os.path.join(output_dir, image_name[:-4] + '_masks.npy'), mask)
    np.save(os.path.join(output_dir, image_name[:-4] + '_boxes.npy'), boxes)
    np.save(os.path.join(output_dir, image_name[:-4] + '_areas.npy'), areas)

def gen_targets(args):
    local_threads = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=args.jobs,
                label='local_threads'
            )
        ]
    )
    local_threads.checkpoint_mode = 'manual'
    run_dir = os.path.join(args.run_dir, args.name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    local_threads.run_dir = run_dir
    parsl.load(local_threads)
    workers = []
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)
        image_root = os.path.join(split_dir, 'images')
        json_root = os.path.join(split_dir, 'json')
        output_dir = os.path.join(split_dir, 'targets')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not os.path.isdir(image_root):
            os.mkdir(image_root)
        for image_name in os.listdir(image_root):
            workers.append(convert_to_targets(image_root, json_root, output_dir, image_name))
    for worker in tqdm(workers):
        result = worker.result()
    parsl.clear()
