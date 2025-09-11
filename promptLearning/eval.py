import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
import deepspeed

from model_utils import load_tokenizer, load_model
from phy_datasets import PhyDataset, load_and_preprocess, collate_fn_left
from transformers import DataCollatorForLanguageModeling
from functools import partial


# ================================================================
# Argument Parser
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=8)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# ================================================================
# Device 설정
# ================================================================
device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(device)

# ================================================================
# Tokenizer / Model 로드
# ================================================================
tokenizer = load_tokenizer()
# model = load_model("Bllossom/llama-3.2-Korean-Bllossom-3B")
# #model.resize_token_embeddings(len(tokenizer))

# ds_engine = deepspeed.init_inference(
#     model,
#     mp_size=1,
#     dtype=torch.float16,
#     checkpoint=args.checkpoint
# )
# model = ds_engine.module

model = load_model(args.checkpoint) 
model.to(device)
model.eval()

# ================================================================
# 데이터 로드
# ================================================================
_, _, test = load_and_preprocess()
test_dataset = PhyDataset(test.iloc[:100], tokenizer, max_length=args.max_length)

# train.py와 동일하게 collator 사용
#collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

collate_fn = partial(collate_fn_left, tokenizer=tokenizer)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
)

# ================================================================
# 모델 generate
# ================================================================
#output_token_id = tokenizer.convert_tokens_to_ids("[OUTPUT]") 
all_pred_texts = []

for batch in tqdm(test_loader, desc="Generating predictions"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=2.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    # for pred_ids in outputs.cpu():
    #     try:
    #         output_pos = (pred_ids == output_token_id).nonzero(as_tuple=True)[0][0].item()
    #         pred_ids = pred_ids[output_pos + 1:]
    #     except:
    #         pass
    #     pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
    #     all_pred_texts.append(pred_text)

    # for i, pred_ids in enumerate(outputs):
    #     prompt_len = (batch['input_ids'][i] != tokenizer.pad_token_id).sum().item()
    #     pred_ids = pred_ids[prompt_len:]  # prompt 제외, output만
    #     pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    #     all_pred_texts.append(pred_text)

    for i, pred_ids in enumerate(outputs):
        # prompt 제외
        prompt_len = (batch['attention_mask'][i] != 0).sum().item() - args.max_new_tokens
        pred_ids = pred_ids[prompt_len:]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        all_pred_texts.append(pred_text)

# ================================================================
# 예측 결과 저장
# ================================================================
pd.DataFrame({"pred_text": all_pred_texts}).to_csv(
    "test_predictions_bllossom_0910.csv", index=False, encoding="utf-8-sig"
)

print(f"Generated {len(all_pred_texts)} predictions. Saved to test_predictions_bllossom_0910.csv")
