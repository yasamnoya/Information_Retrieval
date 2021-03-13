import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="../data/")
    parser.add_argument("-valid_percentage", type=int, default=10)
    args = parser.parse_args()

    queries_df = pd.read_csv(args.data_dir + "queries.csv")
    queries_df = queries_df.reindex(
        np.random.permutation(queries_df.index))

    split_point = int((1 - (args.valid_percentage / 100)) * queries_df.shape[0])
    print(split_point)

    train_queries_df = queries_df.iloc[:split_point]
    valid_queries_df = queries_df.iloc[split_point:]

    train_queries_df.to_csv(args.data_dir + "train_queries.csv")
    valid_queries_df.to_csv(args.data_dir + "valid_queries.csv")

    print(train_queries_df.shape)
    print(valid_queries_df.shape)
