import torch
import os

import node_helpers
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence

def load_image(image_path: str):
    image_path = Path(image_path)
        
    img = node_helpers.pillow(Image.open, image_path)
    
    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']
    
    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]
        
        if image.size[0] != w or image.size[1] != h:
            continue
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def get_all_image_paths(directory):
    # 传入一个目录, 返回该目录下所有图片的路径, 包括子目录
    all_image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_image_paths.append(os.path.join(root, file))
    return all_image_paths


def write_caption_to_txt(caption: str, image_path: str):
    path = os.path.splitext(image_path)[0] + '.txt'

    with open(path, 'w') as f:
        f.write(caption)
    return path