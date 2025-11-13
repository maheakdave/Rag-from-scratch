class Config:
    
    #DATASET
    batch_size = 1
    num_workers = 1
    shuffle=True
    drop_last = True
    document_path = r'.\sample.txt'
    context_length = 5
    stride = 1
    tokenizer_type = 'BPE' # "BPE" | "Word"

    #TRAINING
    

    #MODEL
    GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True       # Query-Key-Value bias
    }
    
    