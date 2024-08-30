from .MultiTextEncode import MultiTextEncode
from .KolorsMultiTextEncode import KolorsMultiTextEncode
from .captioning.Caption import Caption, CaptionDownload
from .batch.BatchImageProcessor import BatchImageProcessor

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MultiTextEncode": MultiTextEncode,
    "KolorsMultiTextEncode": KolorsMultiTextEncode,
    "Caption": Caption,
    "CaptionDownload": CaptionDownload,
    "BatchImageProcessor": BatchImageProcessor
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTextEncode": "MultiTextEncode(批量文本编码)",
    "KolorsMultiTextEncode": "KolorsMultiTextEncode(Kolors批量文本编码)",
    "Caption": "Caption(反推提示词)",
    "CaptionDownload": "CaptionDownload(下载模型)",
    "BatchImageProcessor": "BatchImageProcessor(批量图片处理)"
}