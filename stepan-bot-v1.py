import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelWithLMHead

import torch

import os
import time

SP1 = '@@ПЕРВЫЙ@@'
SP2 = '@@ВТОРОЙ@@'
BASIC_PATH = './'
MODELS = {
    'default': 'tinkoff-ai/ruDialoGPT-medium'
}
MODEL_NAME = MODELS['default']


def get_model_name_or_path():
    if os.path.exists(f'{BASIC_PATH}/training/final/{MODEL_NAME}/default'):
        return f'{BASIC_PATH}/training/final/{MODEL_NAME}/default'
    return MODEL_NAME


def type_slowly(text):
    for c in text:
        print(c, end='')
        time.sleep(0.1)
    print()


CONFIGS = {
    'default': dict(
        top_k=15,
        top_p=0.95,
        num_beams=5,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=2.1,
        repetition_penalty=1.2,
        length_penalty=0.5,
        eos_token_id=50257,
        pad_token_id=0,
        max_new_tokens=40
    ),
    'experiment': dict(
        top_k=5,
        top_p=0.97,
        num_beams=5,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=3.5,
        repetition_penalty=1.2,
        length_penalty=1.2,
        eos_token_id=50257,
        pad_token_id=0,
        max_new_tokens=100
    )
}


def run_chat(model, tokenizer):
    history = f'{SP1} '
    while True:
        inp = input('>> User: ')
        if inp == 'quit':
            break
        if inp == 'restart':
            history = f'{SP1} '
        inputs = tokenizer(history + inp + ' ' + SP2, return_tensors='pt')

        print(f'>> StepanBot:', end=' ')
        generated_token_ids = model.cpu().generate(
            **inputs,
            **CONFIGS['default']
        )

        while len(generated_token_ids[0]) >= 300:
            generated_token_ids = generated_token_ids[:, 100:]

        history = \
            [tokenizer.decode(sample_token_ids, skip_special_tokens=True) for sample_token_ids in generated_token_ids][0]

        type_slowly(history[history.rfind(SP2):].lstrip(SP2).rstrip(SP1).strip())


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelWithLMHead.from_pretrained(get_model_name_or_path())
    run_chat(model, tokenizer)
