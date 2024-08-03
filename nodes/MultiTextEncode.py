import torch

class MultiTextEncode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "clip": ("CLIP", )
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "multi_text_set_area"
    CATEGORY = "Example"

    def multi_text_set_area(self, clip, text, pre_text='',app_text=''):
        # 换行符分割
        text_list = text.split('\n')

        if len(text_list)==0:
            raise Exception('No text found in input')

        pooled_out = []
        cond_out = []
        for i in range(len(text_list)):
            if text_list[i]=='' or text_list[i]=='\n':
                continue
            cond, pooled=self.encode(clip,pre_text+' '+text_list[i]+' '+app_text)
            cond_out.append(cond)
            pooled_out.append(pooled)

        final_pooled_output = torch.cat(pooled_out, dim=0)
        final_conditioning = torch.cat(cond_out, dim=0)

        return ([[final_conditioning, {"pooled_output": final_pooled_output}]],)

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooled


