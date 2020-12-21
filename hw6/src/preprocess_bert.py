import json
import torch
import argparse
from transformers import XLNetTokenizerFast
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx]
                for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def tokenizing_worker(pool_args):
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-large-cased")
    json_data_chunk = pool_args
    doc_text_chunk = [i['doc_text'] for i in json_data_chunk]
    query_text_chunk = [i['query_text'] for i in json_data_chunk]
    label_chunk = [i['tgt'] for i in json_data_chunk]

    encoding_dict = tokenizer(query_text_chunk, doc_text_chunk, padding='max_length',
                              truncation=True, max_length=2048)
    sub_dataset = [{"input_ids": encoding_dict["input_ids"][i],
                    "token_type_ids": encoding_dict["token_type_ids"][i],
                    "attention_mask": encoding_dict["attention_mask"][i],
                    "labels": label_chunk[i]} for i in range(len(label_chunk))]

    return sub_dataset


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument("-from_dir", type=str, default="../json/")
    parser.add_argument("-save_dir", type=str, default="../bert_data/")
    parser.add_argument("-chunk_size", type=int, default=5000)
    args = parser.parse_args()


    with open(args.from_dir + "train.json", 'r') as json_file:
        json_data = json.loads(json_file.read())

        json_data_chunks = [json_data[i:i+args.chunk_size]
                            for i in range(0, len(json_data), args.chunk_size)]

        with Pool(cpu_count()) as p:
            pool_args = json_data_chunks
            i = 0
            for sub_dataset in tqdm(p.imap_unordered(tokenizing_worker, pool_args), total=len(pool_args)):
                torch.save(sub_dataset, f"{args.save_dir}train.{i}.pt")
                i += 1
