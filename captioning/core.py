import torch

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .image import tensor2pil


def get_caption(image: torch.Tensor, prompt: str, max_new_tokens: int, temperature: float, text_model, clip_processor, clip_model, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, image_adapter, device) -> str:

    input_image = tensor2pil(image)
    pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values.to(device=device)
    
    prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

    with torch.amp.autocast_mode.autocast(device=device, enabled=True):
        vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to(device=device)

    prompt_embeds = text_model.model.embed_tokens(prompt.to(device=device))
    assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
    embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        prompt,
    ], dim=1).to(device=device)

    attention_mask = torch.ones_like(input_ids)  

    generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, top_k=10, temperature=temperature, suppress_tokens=None)

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    r = caption.strip()

    return r