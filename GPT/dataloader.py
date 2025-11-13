import torch #type:ignore
from tokenizer import get_tokenizer #type:ignore
from config import Config #type:ignore


class LLMDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer,context_length,stride,document_path):

        with open(document_path,'r') as f:
            txt = f.read()

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids)-context_length,stride):
            input_chunk = token_ids[i:i+context_length]
            target_chunk = token_ids[i+1:i+context_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader(config):
    tokenizer = get_tokenizer(config.tokenizer_type)
    dataset = LLMDataset(tokenizer=tokenizer,context_length=config.context_length,stride=config.stride,document_path=config.document_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        num_workers=config.num_workers
    )

    return dataloader

if __name__ == "__main__":
    cfg = Config()
    dataloader = create_dataloader(config=cfg)
    xb,yb = next(iter(dataloader))
    print(xb.shape)
    print(yb.shape)