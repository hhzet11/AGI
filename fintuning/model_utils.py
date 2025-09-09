# model_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(model_name="Bllossom/llama-3.2-Korean-Bllossom-3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": ["[INPUT]", "[REL]", "[OUTPUT]"]}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def load_model(model_name="Bllossom/llama-3.2-Korean-Bllossom-3B", tokenizer=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    return model
