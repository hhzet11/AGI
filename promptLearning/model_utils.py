# model_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(model_name="Bllossom/llama-3.2-Korean-Bllossom-3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    #special_tokens = {"additional_special_tokens": ["[INPUT]", "[REL]", "[OUTPUT]"]}
    #tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def load_model(model_name="Bllossom/llama-3.2-Korean-Bllossom-3B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    return model
