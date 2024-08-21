from .MultiTextEncode import MultiTextEncode
from .KolorsMultiTextEncode import KolorsMultiTextEncode
from .Caption import Caption, CaptionLoad

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MultiTextEncode": MultiTextEncode,
    "KolorsMultiTextEncode": KolorsMultiTextEncode,
    "Caption": Caption,
    "CaptionLoad": CaptionLoad
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTextEncode": "MultiTextEncode(批量文本编码)",
    "KolorsMultiTextEncode": "KolorsMultiTextEncode(Kolors批量文本编码)",
    "Caption": "Caption(反推提示词)",
    "CaptionLoad": "CaptionLoad(加载模型)"
}