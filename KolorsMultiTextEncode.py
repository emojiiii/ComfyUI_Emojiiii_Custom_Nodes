import torch

from .kolors_core import chatglm3_text_encode

class KolorsMultiTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "encode"
    CATEGORY = "emojiiii"

    def encode(self, chatglm3_model, text):
                # 换行符分割
        text_list = text.split('\n')

        if len(text_list)==0:
            raise Exception('No text found in input')

        pooled_out = []
        cond_out = []
        for i in range(len(text_list)):
            if text_list[i]=='' or text_list[i]=='\n':
                continue
            prompt_embeds, pooled_output = chatglm3_text_encode(self, chatglm3_model, prompt=text_list[i])
            cond_out.append(prompt_embeds)
            pooled_out.append(pooled_output)
        
        final_pooled_output = torch.cat(pooled_out, dim=0)
        final_conditioning = torch.cat(cond_out, dim=0)

        return ([[final_conditioning, {"pooled_output": final_pooled_output}]],)
