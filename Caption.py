class Caption:
    """
    from https://github.com/StartHua/Comfyui_CXH_joy_caption
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "clip": ("CLIP", ),
            }
        }
    


