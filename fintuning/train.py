# train.py
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from konlpy.tag import Okt
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from model_utils import load_model, load_tokenizer
from phy_datasets import PhyDataset, load_and_preprocess
import numpy as np
from torch.optim import AdamW

# ----------------- 토크나이저/모델/데이터셋 -----------------
tokenizer = load_tokenizer()
model = load_model(tokenizer=tokenizer)

# 데이터 로드 및 Dataset 생성
train_df, dev_df, test_df = load_and_preprocess()
train_dataset = PhyDataset(train_df, tokenizer, max_length=16)
dev_dataset = PhyDataset(dev_df, tokenizer, max_length=16)
test_dataset = PhyDataset(test_df, tokenizer, max_length=16)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
okt = Okt()

# ----------------- compute_metrics -----------------
def compute_metrics(eval_pred):
    """
    BLEU-1~4, sacreBLEU, chrF, chrF++, CIDEr, BERTScore, eval_loss 계산
    """
    try:
        predictions, labels = eval_pred

        # tensor 변환
        predictions = torch.tensor(predictions, dtype=torch.float32) if not isinstance(predictions, torch.Tensor) else predictions
        labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

        # Shift for causal LM
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        valid_mask = shift_labels != -100
        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask]

        if len(valid_labels) == 0:
            return {
                "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                "sacrebleu": 0.0, "chrf": 0.0, "chrfpp": 0.0,
                "cider": 0.0, "bertscore": 0.0, "eval_loss": float("inf")
            }

        # eval_loss
        loss = F.cross_entropy(valid_logits, valid_labels)
        eval_loss = loss.item()

        # === Generation for NLG metrics ===
        from transformers import AutoTokenizer
        global tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer must be defined globally for compute_metrics")

        pred_token_ids = torch.argmax(predictions, dim=-1)
        label_token_ids = labels

        pred_texts, label_texts = [], []
        for pred_ids, label_ids in zip(pred_token_ids, label_token_ids):
            label_ids = [i for i in label_ids.tolist() if i != -100 and i != tokenizer.pad_token_id]
            pred_ids = [i for i in pred_ids.tolist() if i != -100 and i != tokenizer.pad_token_id]
            pred_texts.append(tokenizer.decode(pred_ids, skip_special_tokens=True).strip())
            label_texts.append(tokenizer.decode(label_ids, skip_special_tokens=True).strip())

        # BLEU-1~4
        bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
        smoothie = SmoothingFunction().method1
        for ref, hyp in zip(label_texts, pred_texts):
            ref_tokens = [ref.split()]
            hyp_tokens = hyp.split()
            bleu1s.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
            bleu2s.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
            bleu3s.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
            bleu4s.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

        # sacreBLEU (sentence-level)
        sacrebleu_scores = []
        for ref, hyp in zip(label_texts, pred_texts):
            try:
                hyp_tok = ' '.join(okt.morphs(hyp))
                ref_tok = ' '.join(okt.morphs(ref))
                bleu_score = sacrebleu.sentence_bleu(hyp_tok, [ref_tok])
                sacrebleu_scores.append(bleu_score.score)
            except:
                sacrebleu_scores.append(0.0)

        # chrF / chrF++
        try:
            chrf_score = sacrebleu.corpus_chrf(pred_texts, [label_texts], beta=2.0).score
            chrfpp_score = sacrebleu.corpus_chrf(pred_texts, [label_texts], beta=2.0, word_order=2).score
        except:
            chrf_score = 0.0
            chrfpp_score = 0.0

        # CIDEr
        try:
            from pycocoevalcap.cider.cider import Cider
            gts = {i: [label_texts[i]] for i in range(len(label_texts))}
            res = {i: [pred_texts[i]] for i in range(len(pred_texts))}
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
        except:
            cider_score = 0.0

        # BERTScore
        try:
            P, R, F1 = bert_score(pred_texts, label_texts, lang="ko", verbose=False)
            bertscore = float(F1.mean())
        except:
            bertscore = 0.0

        return {
            "bleu1": float(np.mean(bleu1s)),
            "bleu2": float(np.mean(bleu2s)),
            "bleu3": float(np.mean(bleu3s)),
            "bleu4": float(np.mean(bleu4s)),
            "sacrebleu": float(np.mean(sacrebleu_scores)),
            "chrf": float(chrf_score),
            "chrfpp": float(chrfpp_score),
            "cider": float(cider_score),
            "bertscore": float(bertscore),
            "eval_loss": eval_loss,
        }

    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
            "sacrebleu": 0.0, "chrf": 0.0, "chrfpp": 0.0,
            "cider": 0.0, "bertscore": 0.0, "eval_loss": float("inf")
        }

# ----------------- TrainingArguments -----------------
training_args = TrainingArguments(
    output_dir="./outputs",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,
    gradient_accumulation_steps=8,
    deepspeed="deepspeed_config.json",  # <-- DeepSpeed 사용
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)

# ----------------- Trainer -----------------

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# ----------------- 학습 시작 -----------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(training_args.output_dir)
