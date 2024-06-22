import numpy as np
import torch
import re

from torch.utils.data import Dataset, DataLoader

import re

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LEN = 512

def prepare_text(raw_text, tokenizer, max_len=MAX_LEN):
    out_txt = re.sub(r'[^А-я0-9 ();:.,!?\-]+', '', raw_text)
    encoding = tokenizer.encode_plus(
            out_txt,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
    )    
    return out_txt, encoding

class KADDataset(Dataset):

    def __init__(self, texts, filenames, tokenizer, max_len):
        self.texts = texts
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        filename = str(self.filenames[item])
        cleaned_text, encoding = prepare_text(text, self.tokenizer, self.max_len)
        
        return {
            'case_text': text,
            'cleaned_text': cleaned_text,
            'filename': filename,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def create_data_loader(df, tokenizer, max_len=512, batch_size=1):
    ds = KADDataset(
        texts=df.text.to_numpy(),
        filenames=df.filenames.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )
