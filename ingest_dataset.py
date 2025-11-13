import pandas as pd
import re
from GPT.gpt import get_llm
from GPT.tokenizer import get_tokenizer
import chromadb
from multiprocessing import Pool

def clean_text(s):
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def ingest_marco(dataframe:pd.DataFrame):
    client = chromadb.PersistentClient("./MS-MARCO-DB")
    collection = client.get_or_create_collection(name="MS-MARCO")

    df = dataframe.dropna(how='any',axis=0)
    df = df.map(clean_text)

    model = get_llm(category='gpt-2')

    for idx,(answer,query,passage) in df.iterrows():
        embedding = model.get_embedding(text=passage,tokenizer = get_tokenizer("BPE"))
        collection.add(
            embeddings=embedding,
            documents=[passage],
            ids = ["id"+str(idx)]
        )
        print(f'\riter: {idx}/{len(df)}',end='\r')

def partition_dataframe(dataframe: pd.DataFrame, num_partitions: int) -> list[pd.DataFrame]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer")
    n_rows = len(dataframe)
    if n_rows == 0:
        return [dataframe.iloc[0:0].copy() for _ in range(num_partitions)]
    base = n_rows // num_partitions
    rem = n_rows % num_partitions
    parts = []
    start = 0
    for i in range(num_partitions):
        size = base + (1 if i < rem else 0)
        end = start + size
        parts.append(dataframe.iloc[start:end])
        start = end
    return parts

if __name__ == "__main__":
    train_df = pd.read_csv(r'D:\llm\DATASET\train.csv')
    val_df = pd.read_csv(r'D:\llm\DATASET\valid.csv')

    #multiprocessing slower than single process

    # num_partitions = 5
    # train_df_partitioned = partition_dataframe(train_df,num_partitions=num_partitions)
    # with Pool(num_partitions) as p:
    #     p.map(ingest_marco,train_df_partitioned)

    ingest_marco(dataframe=train_df)