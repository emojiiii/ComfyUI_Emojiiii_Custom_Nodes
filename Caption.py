import torch

from captioning.model import snapshot_download, hf_hub_download
import comfy.model_management

class Caption:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "joy_model": ("JOY_MODEL", ),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "max_new_tokens":("INT", {"default": 300, "min": 10, "max": 1000, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    CATEGORY="emojiiii/caption"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate(self, image, joy_model, prompt, max_new_tokens, temperature):
    
        pass


    
class CaptionLoad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Meta-Llama-3.1-8B-bnb-4bit", "meta-llama/Meta-Llama-3.1-8B"], ),
            },
            "optional": {
                "clip": ("CLIP", ),
            }
        }
    
    CATEGORY="emojiiii/caption"
    FUNCTION = "load"
    RETURN_TYPES = ("JOY_MODEL",)

    def load(self, model, clip):

        model_path = snapshot_download(model, 'LLM')

        clip_path = snapshot_download(clip, "clip")

        joy_caption_path = hf_hub_download('fancyfeast/joy-caption-pre-alpha', 'Joy_caption', 'image_adapter.pt')

        return (model_path, clip_path, joy_caption_path)


