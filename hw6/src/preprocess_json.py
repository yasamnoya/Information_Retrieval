import pandas as pd
import argparse
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", type=str, default="../data/")
parser.add_argument("-json_dir", type=str, default="../json/")
args = parser.parse_args()


def preprocess_set(df_doc, df_queries):
    split_point = 250
    pos_pool_args = []
    neg_pool_args = []
    pos_set = []
    neg_set = []

    for _, row in df_queries.iterrows():
        query_text = row.query_text
        doc_id_list = row.bm25_top1000.split(" ")
        pos_id_list = doc_id_list[:split_point]
        neg_id_list = doc_id_list[split_point:]

        pos_pool_args.append((df_doc, query_text, pos_id_list, float(1)))
        neg_pool_args.append((df_doc, query_text, neg_id_list, float(0)))

        # preprocessed_set += get_subset_of_query(df_doc, query_text, pos_id_list, float(1))
        # preprocessed_set += get_subset_of_query(df_doc, query_text, neg_id_list, float(0))

    with Pool(cpu_count()) as p:
        for subset in tqdm(p.imap_unordered(get_subset_of_query, pos_pool_args), total=len(pos_pool_args)):
            pos_set += subset

    with Pool(cpu_count()) as p:
        for subset in tqdm(p.imap_unordered(get_subset_of_query, neg_pool_args), total=len(neg_pool_args)):
            neg_set += subset

    return pos_set, neg_set


def get_subset_of_query(pool_args):  # is_pos = [1,0]
    df_doc, query_text, id_list, is_pos = pool_args
    subset = []

    for doc_id in id_list:
        mask = df_doc.loc[:, "doc_id"] == doc_id
        doc_text = df_doc[mask].doc_text.values[0]

        query_text = str(query_text)
        doc_text = str(doc_text)
        tgt = is_pos

        subset.append({"query_text": query_text,
                       "doc_text": doc_text, "tgt": tgt})

    return subset


if __name__ == "__main__":
    print("Loading...")
    df_doc = pd.read_csv(args.data_dir + "documents.csv")
    df_train_queries = pd.read_csv(args.data_dir + "train_queries.csv")
    df_test_queries = pd.read_csv(args.data_dir + "test_queries.csv")

    print("Preprocessing...")
    train_pos_set, train_neg_set = preprocess_set(df_doc, df_train_queries)
    with open(args.json_dir + "train.pos.json", 'w') as json_file:
        json_file.write(json.dumps(train_pos_set))
    with open(args.json_dir + "train.neg.json", 'w') as json_file:
        json_file.write(json.dumps(train_neg_set))

    test_pos_set, test_neg_set = preprocess_set(df_doc, df_test_queries)
    with open(args.json_dir + "test.pos.json", 'w') as json_file:
        json_file.write(json.dumps(test_pos_set))
    with open(args.json_dir + "test.neg.json", 'w') as json_file:
        json_file.write(json.dumps(test_neg_set))
