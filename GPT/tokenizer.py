import re
import tiktoken # type:ignore



def create_vocab(sample:str)->dict[str,int]:
    split = re.split(r'[.,:;!?"/()\']|--|\s',sample)
    preprocessed = [item.strip() if item.strip() else item for item in split]
    vocab_list = list(set(preprocessed))
    vocab_list.extend(["<|endoftext|>","<|unk|>"])
    vocab:dict[str,int] = {word:i for i,word in enumerate(vocab_list)}
    return vocab

class WordTokenizer:
    def __init__(self,vocab:dict[str,int])->None:
        self.str_to_id:dict = vocab
        self.id_to_str:dict = {idx:word for word,idx in vocab.items()}
    
    def encode(self,sample:str)->list[int]:
        split = re.split(r'[.,:;!?"/()\']|--|\s',sample)
        tokens = [token.strip() if token.strip() else token for token in split]
        token_ids = [self.str_to_id[token.strip()] if token in self.str_to_id else self.str_to_id["<|unk|>"] for token in tokens]
        return token_ids
    
    def decode(self,sample:list[int])->str:
        txt = " ".join([self.id_to_str[id] for id in sample])
        txt = re.sub(r'\s+([,.:;?!"()\'])',r'\1',txt)
        return txt

def get_tokenizer(tokenizer_type:str):
    if tokenizer_type == 'BPE':
        return tiktoken.get_encoding('gpt2')
    
    

if __name__ == '__main__':

    with open('sample.txt') as f:
        document = f.read()

    vocab = create_vocab(document)


    print(len(vocab))
    print(document[:100])

    print('Word tokenizer from scratch:')

    tokenizer = WordTokenizer(vocab)
    print(tokenizer.encode(document[:100]))
    print(tokenizer.decode(tokenizer.encode(document[:100])))

    
    print("Byte-Pair tokenizer:")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
    )

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    print(integers)
    strings = tokenizer.decode(integers)

    print(strings)