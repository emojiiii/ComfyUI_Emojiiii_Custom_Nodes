import torch
import os
import random
import re
import gc
import json
import comfy.model_management as mm

def chatglm3_text_encode(self, chatglm3_model, prompt):
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    mm.unload_all_models()
    mm.soft_empty_cache()
        # Function to randomly select an option from the brackets
    def choose_random_option(match):
        options = match.group(1).split('|')
        return random.choice(options)

    prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)

    # Define tokenizers and text encoders
    tokenizer = chatglm3_model['tokenizer']
    text_encoder = chatglm3_model['text_encoder']
    text_encoder.to(device)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    output = text_encoder(
            input_ids=text_inputs['input_ids'] ,
            attention_mask=text_inputs['attention_mask'],
            position_ids=text_inputs['position_ids'],
            output_hidden_states=True)
    
    prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
    text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    bs_embed = text_proj.shape[0]
    text_proj = text_proj.repeat(1, 1).view(
        bs_embed, -1
    )
    text_encoder.to(offload_device)
    mm.soft_empty_cache()
    gc.collect()    
    return prompt_embeds, text_proj