import json
import math
import torch
import argparse
from transformers import RobertaTokenizerFast
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def tokenizing_worker(pool_args):
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    json_data_chunk = pool_args
    doc_text_chunk = [i['doc_text'] for i in json_data_chunk]
    query_text_chunk = [i['query_text'] for i in json_data_chunk]
    label_chunk = [i['label'] for i in json_data_chunk]

    encoding_dict = tokenizer(query_text_chunk, doc_text_chunk, padding='max_length',
                              truncation=True, max_length=512)
    sub_dataset = [{"input_ids": encoding_dict["input_ids"][i],
                    # "token_type_ids": encoding_dict["token_type_ids"][i],
                    "token_type_ids": [0] * 512,
                    "attention_mask": encoding_dict["attention_mask"][i],
                    "labels": label_chunk[i]} for i in range(len(label_chunk))]

    return sub_dataset


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument("-from_dir", type=str, default="../json/")
    parser.add_argument("-save_dir", type=str, default="../bert_data/")
    parser.add_argument("-num_splits", type=int, default=12)
    args = parser.parse_args()

    for pos_neg in ["pos", "neg"]:
        with open(args.from_dir + f"train.{pos_neg}.json", 'r') as json_file:
            json_data = json.loads(json_file.read())
            chunk_size = math.ceil(len(json_data) / (args.num_splits))

            json_data_chunks = [json_data[i:i + chunk_size]
                                for i in range(0, len(json_data), chunk_size)]

            with Pool(cpu_count()) as p:
                pool_args = json_data_chunks
                for i, sub_dataset in tqdm(enumerate(p.imap_unordered(tokenizing_worker, pool_args)), total=len(pool_args)):
                    torch.save(
                            sub_dataset, f"{args.save_dir}train.{pos_neg}.{i}.pt")
