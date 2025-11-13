import torch #type:ignore
from .config import Config
from .tokenizer import get_tokenizer
import time
import os
from safetensors.torch import load_file
import requests
import glob
import logging



class SelfAttention(torch.nn.Module):

    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys =  self.W_key(x)
        queries =  self.W_query(x)
        values =  self.W_value(x)
        
        attn_scores = queries @ keys.mT 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

class CausalSelfAttention(torch.nn.Module):
    def __init__(self,embed_dim_in,embed_dim_out,context_length,dropout_prob=0.5,qkv_bias = False):
        super().__init__()
        self.W_key = torch.nn.Linear(embed_dim_in,embed_dim_out,bias=qkv_bias)
        self.W_query = torch.nn.Linear(embed_dim_in,embed_dim_out,bias=qkv_bias)
        self.W_value = torch.nn.Linear(embed_dim_in,embed_dim_out,bias=qkv_bias)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self,x):
        _,num_tokens,embed_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        omega =   queries @ keys.transpose(1,2)
        omega.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        omega = torch.softmax(omega/(embed_dim**0.5),dim=-1)
        omega = self.dropout(omega)
        context_vector = omega@values
        return context_vector

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self,embed_dim_in,embed_dim_out,num_heads,dropout,qkv_bias=False):
        super().__init__()
        assert embed_dim_out%num_heads == 0, "Number of heads should be multiple of embed_dim_out"
        self.num_heads = num_heads
        self.head_dim = embed_dim_out//num_heads
        self.embed_dim_out = embed_dim_out
        self.qkv = torch.nn.Linear(embed_dim_in,embed_dim_out*3,bias=qkv_bias)
        self.dropout = dropout
        self.out_proj = torch.nn.Linear(embed_dim_out,embed_dim_out,bias=qkv_bias)
    def forward(self,x):
        b,num_tokens,_ = x.shape
        qkv = self.qkv(x).view(b,num_tokens,3,self.num_heads,self.head_dim)
        q,k,v = torch.einsum('bnahd->abhnd',qkv)
        
        use_dropout = 0. if not self.training else self.dropout

        #Uses Flash attention internally
        context_vector = torch.nn.functional.scaled_dot_product_attention(query=q,key=k,value=v,dropout_p=use_dropout,is_causal=True)

        return self.out_proj(context_vector.transpose(1,2).contiguous().view(b,num_tokens,self.embed_dim_out))

class FeedForward(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            torch.nn.GELU(),
            torch.nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )

    def forward(self,x):
        return self.layers(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiheadSelfAttention(embed_dim_in=cfg["emb_dim"],embed_dim_out=cfg["emb_dim"],num_heads=cfg["n_heads"],dropout=cfg["drop_rate"],qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = torch.nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = torch.nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x+shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut
        return x

class GPT2(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_embedding = torch.nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_embedding = torch.nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ) # (*[TransformerBlock(cfg)]*cfg["n_layers"]) causes same weight init accross all blocks.
        self.final_norm = torch.nn.LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        self.apply(self.init_weights)

    def init_weights(self,module):
            if isinstance(module,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif isinstance(module,torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,tokens):
        _,seq_len = tokens.shape
        token_embeddings = self.tok_embedding(tokens)
        pos_embedding = self.pos_embedding(torch.arange(seq_len,device=tokens.device))

        x = token_embeddings+pos_embedding
        x = self.dropout(x)

        for blck in self.trf_blocks:
            x = blck(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    @torch.no_grad()
    def get_embedding(self,text,tokenizer):
        encoded = torch.tensor(tokenizer.encode(text)).unsqueeze(0).cuda()
        _,seq_len = encoded.shape
        tok_embedding = self.tok_embedding(encoded)
        pos_embedding = self.pos_embedding(torch.arange(seq_len).cuda())
        x = tok_embedding+pos_embedding
        for blck in self.trf_blocks:
            x = blck(x)
        x = self.final_norm(x)
        pooled = x.mean(dim=1).squeeze(0)                                      
        pooled = pooled / pooled.norm(p=2).clamp(min=1e-12)                    
        return pooled.cpu().tolist()

@torch.no_grad()
def generate_with_greedy_simple(model,tokens,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        stripped_tokens = tokens[:,-context_size:]
        logits = model(stripped_tokens)[:,-1,:]
        token_next = torch.argmax(torch.log_softmax(logits,dim=-1),dim=-1,keepdim=True)
        tokens = torch.cat((tokens,token_next),dim=1)
    return tokens

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())

def download_weights_file(CHOOSEN_MODEL):

    URL_DIR = {
        "gpt2-small (124M)": "gpt2",         # works ok
        "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
        "gpt2-large (774M)": "gpt2-large",   # works ok
        "gpt2-xl (1558M)": "gpt2-xl"         # works ok
    }
    default = "gpt2-small (124M)"
    if CHOOSEN_MODEL is None:
        CHOOSEN_MODEL = default
    url = f"https://huggingface.co/openai-community/{URL_DIR[CHOOSEN_MODEL]}/resolve/main/model.safetensors"
    output_file = f"model-{URL_DIR[CHOOSEN_MODEL]}.safetensors"

    if not os.path.exists(output_file):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            f.write(response.content)
            
def load_weights_into_gpt(gpt, params):
    

    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params["wpe.weight"])
    gpt.tok_embedding.weight = assign(gpt.tok_embedding.weight, params["wte.weight"])

    for b in range(len(gpt.trf_blocks)):

        q_w,k_w,v_w = torch.chunk(
            params[f"h.{b}.attn.c_attn.weight"],3,axis=-1
        )
        
        qkv_w = torch.cat((q_w,k_w,v_w),dim=-1)
        qkv_w = params[f"h.{b}.attn.c_attn.weight"].T
        gpt.trf_blocks[b].att.qkv.weight = assign(
            gpt.trf_blocks[b].att.qkv.weight,qkv_w
        )

        
        q_b,k_b,v_b = torch.chunk(
            params[f"h.{b}.attn.c_attn.bias"],3,axis=-1
        )
        
        qkv_b = torch.cat((q_b,k_b,v_b),dim=-1)
        gpt.trf_blocks[b].att.qkv.bias = assign(
            gpt.trf_blocks[b].att.qkv.bias,qkv_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight,
            params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias,
            params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight,
            params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias,
            params[f"h.{b}.ln_2.bias"])

    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["ln_f.weight"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])

def sample_from_logits(logits, temperature=1.0, top_k=50, top_p=0.0):
    logits = logits / max(1e-8, temperature)
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return indices[choice].item()
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = (cumulative > top_p).float().argmax().item()
        cutoff = max(1, cutoff)
        idxs = sorted_idx[: cutoff+1]
        vals = sorted_logits[: cutoff+1]
        probs2 = torch.softmax(vals, dim=-1)
        choice = torch.multinomial(probs2, num_samples=1)
        return idxs[choice].item()
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

@torch.no_grad()
def generate_with_greedy_typical(model, tokens, max_new_tokens, context_size,
                  temperature=0.9, top_k=40, top_p=0.0, repetition_penalty=1.2, eos_token_id=None,stream_len = None):
    generated = tokens.clone()
    seen = []
    if stream_len != None:
        streams = []
        counter = 0
    for i in range(max_new_tokens):
        stripped = generated[:, -context_size:]
        logits = model(stripped)[:, -1, :].squeeze(0)
        if repetition_penalty != 1.0 and len(seen) > 0:
            for t in set(seen[-200:]):
                if logits[t] < 0:
                    logits[t] = logits[t] * repetition_penalty
                else:
                    logits[t] = logits[t] / repetition_penalty

        next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated = torch.cat((generated, torch.tensor([[next_id]], device=generated.device)), dim=1)
        seen.append(next_id)
        if stream_len!=None:
            streams.append(next_id)
            counter+=1
            if counter==stream_len:
                yield streams
                streams.clear()
                counter = 0
        if eos_token_id is not None and next_id == eos_token_id:
            break
        if len(seen) >= 20 and len(set(seen[-20:])) == 1:
            logging.INFO("Breaking due to repetition of a single token.")
            break
        
    return generated

def get_llm(category):
    config = Config()
    if category == 'gpt-2':
        download_weights_file(CHOOSEN_MODEL=None) # will use default 124M model
        state_dict_path = glob.glob("./*.safetensors")[0]
        state_dict = load_file(state_dict_path)

        model = GPT2(config.GPT_CONFIG_124M)
        model.eval()

        load_weights_into_gpt(gpt=model,params=state_dict)
        model = model.to('cuda:0')
        return model



if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,filename="log.log",filemode="w",
                             format = "%(asctime)s - %(levelname)s - %(message)s")
    config = Config()

    download_weights_file(CHOOSEN_MODEL=None) # will use default 124M model
    state_dict_path = glob.glob("./*.safetensors")[0]
    state_dict = load_file(state_dict_path)

    model = GPT2(config.GPT_CONFIG_124M)
    model.eval()

    load_weights_into_gpt(gpt=model,params=state_dict)
    model = model.to('cuda:0')
    start_text = "Once upon a time"

    tokenizer = get_tokenizer(tokenizer_type=config.tokenizer_type)
    encoded = tokenizer.encode(start_text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).cuda()

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_text)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    #Repition of words/sequences
    start = time.time()
    token_ids = generate_with_greedy_simple(
        model=model,
        tokens=encoded_tensor,
        max_new_tokens=200,
        context_size=config.GPT_CONFIG_124M["context_length"]
    )

    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    
    print("Output text:", decoded_text)
    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

    start = time.time()
    token_ids = generate_with_greedy_typical(model=model,tokens=encoded_tensor,max_new_tokens=200,
                                             context_size=config.GPT_CONFIG_124M["context_length"])
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("Output text:", decoded_text)
    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")