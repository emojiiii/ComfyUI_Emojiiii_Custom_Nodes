import torch
import os

from transformers import AutoProcessor, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from .captioning.model import snapshot_download, hf_hub_download
from .captioning.image import load_image
from .files.file import get_all_image_paths, write_caption_to_txt
from .captioning.core import get_caption
import comfy.model_management

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class ImageAdapter(torch.nn.Module):
	def __init__(self, input_features: int, output_features: int):
		super().__init__()
		self.linear1 = torch.nn.Linear(input_features, output_features)
		self.activation = torch.nn.GELU()
		self.linear2 = torch.nn.Linear(output_features, output_features)
	
	def forward(self, vision_outputs: torch.Tensor):
		x = self.linear1(vision_outputs)
		x = self.activation(x)
		x = self.linear2(x)
		return x

class Caption:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_dir": ("STRING", {"multiline": False, "default": ""}),
                "joy_model": ("JOY_MODEL", ),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "max_new_tokens":("INT", {"default": 300, "min": 10, "max": 1000, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "default_image_dir": (AlwaysEqualProxy("*"), {})
            }
        }
    
    CATEGORY="emojiiii/caption"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate(self, image_dir: str, prompt: str, max_new_tokens: int, temperature: float, default_image_dir: str, **kwargs):
        if image_dir == "" or image_dir == None:
            image_dir = default_image_dir

        if not os.path.exists(image_dir):
            raise Exception("Image path does not exist")
       
        device = comfy.model_management.get_torch_device()

        joy_model = kwargs.get("joy_model")
        clip_path = joy_model.get("clip_path")
        model_path = joy_model.get("model_path")
        adapter_path = joy_model.get("joy_caption_path")

        clip_processor = AutoProcessor.from_pretrained(clip_path) 
        clip_model = AutoModel.from_pretrained(clip_path,trust_remote_code=True)
        clip_model = clip_model.vision_model
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        text_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",trust_remote_code=True)
        text_model.eval()

        image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
        adjusted_adapter =  image_adapter #AdjustedImageAdapter(image_adapter, text_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to(device=device)

        text = ''
        if os.path.isdir(image_dir):
            all_image_paths = get_all_image_paths(image_dir)
            
            for image_path in all_image_paths:
                image_tr, mask = load_image(image_path)
                temp = get_caption(image=image_tr,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        text_model=text_model,
                        clip_processor=clip_processor,
                        clip_model=clip_model,
                        tokenizer=tokenizer,
                        image_adapter=adjusted_adapter,
                        device=device
                        )
                write_caption_to_txt(temp, image_path)
                text += temp + '\n\n'
                print(f"Caption for {image_path} is: {temp}")
        else:
            image_tr, mask = load_image(image_path)
            text = get_caption(image=image_tr,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    text_model=text_model,
                    clip_processor=clip_processor,
                    clip_model=clip_model,
                    tokenizer=tokenizer,
                    image_adapter=adjusted_adapter,
                    device=device
                    )

        return (text, )

    
class CaptionDownload:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Meta-Llama-3.1-8B-bnb-4bit", "meta-llama/Meta-Llama-3.1-8B"], ),
                "siglip_model": (["google/siglip-so400m-patch14-384", "google/siglip-so400m-patch14-224"], ),
            }
        }
    
    CATEGORY="emojiiii/caption"
    FUNCTION = "download"
    RETURN_TYPES = ("JOY_MODEL",)

    def download(self, model: str, siglip_model: str):

        model_path = snapshot_download(model, 'LLM')

        clip_path = snapshot_download(siglip_model, "clip")

        joy_caption_path = hf_hub_download(repo_id="fancyfeast/joy-caption-pre-alpha",
                                           filename="image_adapter.pt",
                                           prefix="JoyCaption",
                                           subfolder="wpkklhc6",
                                           repo_type="space"
                                           )

        return ({"model_path": model_path, "clip_path": clip_path, "joy_caption_path": joy_caption_path},)


