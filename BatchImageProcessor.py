import os
from .batch.image import crop_image
from .files.file import get_all_image_paths

class BatchImageProcessor:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_dir": ("STRING", {"multiline": False, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 10, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 10, "step": 1}),
                "auto_face": ("BOOLEAN", {"default": False}),
                "quality": ("INT", {"default": 95, "min": 60, "max": 100, "step": 1}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
                "format": (["jpeg", "png", "jpg"], {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY="emojiiii/caption"
    FUNCTION = "batch_image_process"

    def batch_image_process(self, image_dir, width, height, auto_face, quality, output_dir, format):
        
        if output_dir == "" or output_dir == None:
            output_dir = os.path.join(os.path.dirname(image_dir), "output")

        # Check if the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check if the input directory exists
        if not os.path.exists(image_dir):
            # throw an error if the input directory does not exist
            raise Exception("Input directory does not exist")
        
        # Check image_dir is a directory or a file
        if os.path.isfile(image_dir):
            # Check image_dir is an image file
            if not image_dir.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                raise Exception("Input file is not an image file")
            
            # Crop the image
            crop_image(image_path=image_dir, output_path=output_dir, width=width, height=height, crop_face=auto_face, format=format, quality=quality)
            return (os.path.dirname(output_dir),)
        
        # Get the list of images in the input directory
        images = os.listdir(image_dir)
        images = get_all_image_paths(image_dir)

        for image in images:
            image_path = os.path.join(image_dir, image)
            output_path = os.path.join(output_dir, os.path.basename(image))
            crop_image(image_path=image_path, output_path=output_path, width=width, height=height, crop_face=auto_face, format=format, quality=quality)
        
        return (output_dir,)

