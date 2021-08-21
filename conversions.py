import json
import os

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
        



