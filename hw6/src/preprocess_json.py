import pandas as pd
import random
import argparse
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def preprocess_set(df_doc, df_queries, split_point):
    pos_pool_args = []
    neg_pool_args = []
    pos_set = []
    neg_set = []

    for _, row in df_queries.iterrows():
        query_text = row.query_text
        pos_doc_id_list = row.pos_doc_ids.split(" ")
        bm25_doc_id_list = row.bm25_top1000.split(" ")
        pos_doc_id_list += bm25_doc_id_list[:split_point]
        neg_doc_id_list = [
            i for i in bm25_doc_id_list[split_point:] if i not in pos_doc_id_list]

        pos_pool_args.append((df_doc, query_text, pos_doc_id_list, float(1)))
        neg_pool_args.append((df_doc, query_text, neg_doc_id_list, float(0)))

    for pool_args in tqdm(pos_pool_args):
        pos_set += get_subset_of_query(pool_args)

    for pool_args in tqdm(neg_pool_args):
        neg_set += get_subset_of_query(pool_args)

        random.shuffle(pos_set)
        random.shuffle(neg_set)

    return pos_set, neg_set


def get_subset_of_query(pool_args):  # label = [1,0]
    df_doc, query_text, id_list, label = pool_args
    mask = df_doc.loc[:, "doc_id"].isin(id_list)
    df_doc = df_doc[mask]

    pool_args = []
    for doc_id in id_list:
        mask = df_doc.loc[:, "doc_id"] == doc_id
        doc_text = df_doc[mask].doc_text.values[0]

        query_text = str(query_text)
        doc_text = str(doc_text)
        labkl = int(label)

        pool_args.append((df_doc, query_text, doc_id, label))

    with Pool(cpu_count()) as p:
        subset = p.map(_get_subset_of_query, pool_args)

    return subset


def _get_subset_of_query(pool_args):
    df_doc, query_text, doc_id, label = pool_args
    mask = df_doc.loc[:, "doc_id"] == doc_id
    doc_text = df_doc[mask].doc_text.values[0]

    query_text = str(query_text)
    doc_text = str(doc_text)

    return {"query_text": query_text, "doc_text": doc_text, "label": label}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="../data/")
    parser.add_argument("-json_dir", type=str, default="../json/")
    parser.add_argument("-split_point", type=int, default="20")
    args = parser.parse_args()

    print("Loading...")
    df_doc = pd.read_csv(args.data_dir + "documents.csv")
    df_train_queries = pd.read_csv(args.data_dir + "train_queries.csv")
    df_test_queries = pd.read_csv(args.data_dir + "test_queries.csv")

    print("Preprocessing...")
    train_pos_set, train_neg_set = preprocess_set(
        df_doc, df_train_queries, args.split_point)
    with open(args.json_dir + "train.pos.json", 'w') as json_file:
        json_file.write(json.dumps(train_pos_set))
    with open(args.json_dir + "train.neg.json", 'w') as json_file:
        json_file.write(json.dumps(train_neg_set))
