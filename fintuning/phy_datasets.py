# datasets.py
import pandas as pd
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


class PhyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_token_id = tokenizer.convert_tokens_to_ids("[OUTPUT]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"[INPUT] {row['input_ko']} [REL] {row['relation_ko']} [OUTPUT] {row['output_ko']}"
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        output_pos = (labels == self.output_token_id).nonzero(as_tuple=True)[0]
        if len(output_pos) > 0:
            start = output_pos[0].item()
            labels[: start + 1] = -100
        else:
            labels[:] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
