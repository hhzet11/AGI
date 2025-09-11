# datasets.py
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_and_preprocess():
    # --- CSV 로드 ---
    train = pd.read_csv('/home/apitrain2/hy_agi/phyEn2Ko/0902.tsv', sep='\t', header=0,
                        names=['input', 'relation', 'output', 'sentence', 'vera_score', 'input_ko', 'output_ko', 'relation_ko'])
    dev = pd.read_csv('/home/apitrain2/hy_agi/phyEn2Ko/dev_ko.tsv', sep='\t', header=0,
                      names=['input', 'relation', 'output', 'sentence', 'vera_score', 'input_ko', 'output_ko', 'relation_ko'])
    test = pd.read_csv('/home/apitrain2/hy_agi/phyEn2Ko/test_ko.tsv', sep='\t', header=0,
                       names=['input', 'relation', 'output', 'input_ko', 'output_ko', 'relation_ko'])

    # --- 필요한 컬럼만 선택 ---
    train = train[["input_ko", "relation_ko", "output_ko"]]
    dev   = dev[["input_ko", "relation_ko", "output_ko"]]
    test  = test[["input_ko", "relation_ko", "output_ko"]]


    # --- 중복 제거 함수 ---
    def drop_duplicates_and_report(df, name):
        before = len(df)
        df_dedup = df.drop_duplicates(subset=["input_ko", "relation_ko", "output_ko"])
        after = len(df_dedup)
        print(f"{name}에서 중복 제거: {before - after}개 삭제 (최종 {after}개)")
        return df_dedup

    train = drop_duplicates_and_report(train, "train")
    dev   = drop_duplicates_and_report(dev, "dev")
    test  = drop_duplicates_and_report(test, "test")

    return train, dev, test

def collate_fn_left(batch, tokenizer):
    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    max_len = max(len(b['input_ids']) for b in batch)

    for b in batch:
        ids = b['input_ids']
        attn = b['attention_mask']
        pad_len = max_len - len(ids)

        # left padding
        input_ids_padded.append(
            torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), ids])
        )
        attention_mask_padded.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), attn])
        )
        # labels: padding 부분 -100
        labels_padded.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), ids])
        )

    return {
        'input_ids': torch.stack(input_ids_padded),
        'attention_mask': torch.stack(attention_mask_padded),
        'labels': torch.stack(labels_padded)
    }



class PhyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        #self.output_token_id = tokenizer.convert_tokens_to_ids("[OUTPUT]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- instruct-style prompt ---
        system_prompt = "너는 물리 상식을 알려주는 한국어 어시스턴트야. 답변은 짧고 명확한 명사구로 해줘."
        user_prompt = f"사용자: '{row['input_ko']}'의 '{row['relation_ko']}'는 무엇일까?"
        assistant_prompt = "어시스턴트:"

        full_prompt = f"{system_prompt}\n{user_prompt}\n{assistant_prompt} "
        
        # output은 별도
        output_text = row['output_ko'] + self.tokenizer.eos_token

        # 토크나이징
        tokenized_prompt = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False
        )
        tokenized_output = self.tokenizer(
            output_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False
        )

        # input_ids = prompt + output
        input_ids = torch.cat([tokenized_prompt["input_ids"], tokenized_output["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([tokenized_prompt["attention_mask"], tokenized_output["attention_mask"]], dim=1).squeeze(0)

        # labels = -100 for prompt, output for loss
        labels = torch.cat([
            torch.full_like(tokenized_prompt["input_ids"], -100),
            tokenized_output["input_ids"]
        ], dim=1).squeeze(0)

        # max_length 맞춰 left-padding 적용
        pad_len = self.max_length - input_ids.size(0)
        if pad_len > 0:
            input_ids = torch.cat([
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                input_ids
            ], dim=0)
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                attention_mask
            ], dim=0)
            labels = torch.cat([
                torch.full((pad_len,), -100, dtype=torch.long),
                labels
            ], dim=0)
        else:
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            labels = labels[-self.max_length:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]            

    #     # --- instruct 스타일 바꾸면서 여기를 변경함 ---

    #     #text = f"[INPUT] {row['input_ko']} [REL] {row['relation_ko']} [OUTPUT] {row['output_ko']}"
    #     #encodings = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
    #     #input_ids = encodings["input_ids"].squeeze(0)
    #     #attention_mask = encodings["attention_mask"].squeeze(0)
    #     # labels = input_ids.clone()
    #     # output_pos = (labels == self.output_token_id).nonzero(as_tuple=True)[0]
    #     # if len(output_pos) > 0:
    #     #     start = output_pos[0].item()
    #     #     labels[: start + 1] = -100
    #     # else:
    #     #     labels[:] = -100

    #     # instruct-style prompt
    #     system_prompt = "너는 물리 상식을 알려주는 한국어 어시스턴트야. 답변은 짧고 명확한 명사구로 해줘."
    #     user_prompt = f"사용자: '{row['input_ko']}'의 '{row['relation_ko']}'는 무엇일까?"
    #     assistant_prompt = "어시스턴트:"

    #     full_prompt = f"{system_prompt}\n{user_prompt}\n{assistant_prompt} "
    #     full_text = full_prompt + row['output_ko'] + self.tokenizer.eos_token

    #     # 토크나이징
    #     tokenized_full = self.tokenizer(
    #         full_text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,
    #         return_tensors="pt",
    #         pad_to_multiple_of=None,  # 필요시
    #     )

    #     tokenized_prompt = self.tokenizer(
    #         full_prompt,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,
    #         return_tensors="pt",
    #     )
    #     input_ids = tokenized_full["input_ids"].squeeze(0)
    #     attention_mask = tokenized_full["attention_mask"].squeeze(0)
    #     labels = input_ids.clone()

    #     # 프롬프트 부분은 -100으로 마스킹
    #     prompt_len = (tokenized_prompt["attention_mask"].squeeze(0) == 1).sum().item()
    #     labels[:prompt_len] = -100

    #     # padding side 확인
    #     if tokenized_full['input_ids'][0,0] != self.tokenizer.pad_token_id:
    #         # 직접 left-padding
    #         pad_len = self.max_length - tokenized_full['input_ids'].size(1)
    #         tokenized_full['input_ids'] = torch.cat([
    #             torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long),
    #             tokenized_full['input_ids']
    #         ], dim=1)
    #         tokenized_full['attention_mask'] = torch.cat([
    #             torch.zeros(1, pad_len, dtype=torch.long),
    #             tokenized_full['attention_mask']
    #         ], dim=1)

    #     # ------------------------------------------------

    #     return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
