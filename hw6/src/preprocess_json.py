import pandas as pd
import argparse
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("-doc_path", type=str)
parser.add_argument("-train_path", type=str)
parser.add_argument("-test_path", type=str)
parser.add_argument("-save_train_path", type=str)
parser.add_argument("-save_test_path", type=str)
args = parser.parse_args()


def preprocess_set(df_doc, df_queries):
    split_point = 500
    pool_args = []
    preprocessed_set = []

    for _, row in df_queries.iterrows():
        query_text = row.query_text
        doc_id_list = row.bm25_top1000.split(" ")
        pos_id_list = doc_id_list[:split_point]
        neg_id_list = doc_id_list[split_point:]

        pool_args.append((df_doc, query_text, pos_id_list, float(1)))
        pool_args.append((df_doc, query_text, neg_id_list, float(0)))

        # preprocessed_set += get_subset_of_query(df_doc, query_text, pos_id_list, float(1))
        # preprocessed_set += get_subset_of_query(df_doc, query_text, neg_id_list, float(0))

    with Pool(cpu_count()) as p:
        for subset in tqdm(p.imap_unordered(get_subset_of_query, pool_args), total=len(pool_args)):
            preprocessed_set += subset

    return preprocessed_set


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
    df_doc = pd.read_csv(args.doc_path)
    df_train_queries = pd.read_csv(args.train_path)
    df_test_queries = pd.read_csv(args.test_path)

    print("Preprocessing...")
    train_set = preprocess_set(df_doc, df_train_queries)
    with open(args.save_train_path, 'w') as train_file:
        train_file.write(json.dumps(train_set))

    test_set = preprocess_set(df_doc, df_test_queries)
    with open(args.save_test_path, 'w') as test_file:
        test_file.write(json.dumps(test_set))

