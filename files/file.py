import os

def get_all_image_paths(directory, extension=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    # 传入一个目录, 返回该目录下所有图片的路径, 包括子目录
    all_image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                all_image_paths.append(os.path.join(root, file))
    return all_image_paths

def write_caption_to_txt(caption: str, image_path: str):
    path = os.path.splitext(image_path)[0] + '.txt'

    with open(path, 'w') as f:
        f.write(caption)
    return path