import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
import json

import argparse
import logging
from typing import Union
from tqdm import tqdm

eval_logger = logging.getLogger("eval_3d")

from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle

from eval.dataset.clothocaption import ClothoCapsDataset
from eval.utils import DEFAULT_POINT_TOKEN

def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path='checkpoints/final_result1/audio_finetune_1epoch',
    model_base=None,
    model_name='eagle'
)
model.eval()

modality = 'audio'
test_dataset = ClothoCapsDataset()
test_dataloader = DataLoader(
    test_dataset,
    collate_fn=test_dataset.collate_fn
)

outputs = []
for i, data in enumerate(test_dataloader):
    audio_file, audio_name, audio_id = data[0]
    print(data)

    image_tensor = image_processor(audio_file, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(dtype=torch.float16, device='cuda')

    # question = 'Provide a one-sentence caption for the provided audio.'
    question = "Is this a sound of a car? \nA.YES\nB.NO\n"

    if DEFAULT_IMAGE_TOKEN not in question:
        # question = DEFAULT_IMAGE_TOKEN + '\n' + question
        question = DEFAULT_IMAGE_TOKEN + question
    
    conv = conv_templates['llama3'].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    print(prompt_question)

    # prompt_question = question

    input_ids = tokenizer_image_token(
        prompt_question, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    )
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # print(input_ids)
    input_ids = pad_sequence(
        tokenizer=tokenizer,
        input_ids=[input_ids], 
        batch_first=True, 
        padding_value=pad_token_ids
    ).to('cuda')
    attention_masks = input_ids.ne(pad_token_ids).to('cuda')

    gen_kwargs = {}
    if "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 1024
    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    # try:
    cont = model.generate(
        input_ids,
        attention_mask=attention_masks,
        pad_token_id=pad_token_ids,
        images=image_tensor,
        do_sample=True if gen_kwargs["temperature"] > 0 else False,
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        num_beams=gen_kwargs["num_beams"],
        max_new_tokens=gen_kwargs["max_new_tokens"],
        use_cache=None,
        modality=modality,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # print(text_outputs)
    # except Exception as e:
    #     eval_logger.error(f"Error {e} in generating")
    #     cont = ""
    #     text_outputs = [""]
    outputs.append({
        'image_id': audio_id,
        'caption': text_outputs[0].strip()
    })
    print(outputs)
    break

