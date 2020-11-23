from tqdm import tqdm
from math import log
from multiprocessing import Pool, cpu_count
from sys import argv
import numpy as np

import os


NUM_TOPIC = 4


data_dir = '../ntust-ir-2020_hw4_v2'
OUT_PATH = '../result.csv'



def get_data(data_dir):

    doc_list_path = data_dir + "/doc_list.txt"
    query_list_path = data_dir + "/query_list.txt"

    with open(doc_list_path, 'r') as doc_list_file:
        data = doc_list_file.read().strip()
        doc_filename_list = [line for line in data.split('\n')]
    pass

    with open(query_list_path, 'r') as query_list_file:
        data = query_list_file.read().strip()
        query_filename_list = [line for line in data.split('\n')]
    pass

    doc_list = []
    for filename in doc_filename_list:
        path = data_dir + '/docs/' + filename + '.txt'
        doc = open(path , 'r').read().strip().split(' ')
        doc_list.append(doc)
    pass

    query_list = []
    for filename in query_filename_list:
        path = data_dir + '/queries/' + filename + '.txt'
        query = open(path , 'r').read().strip().split(' ')
        query_list.append(query)
    pass

    return doc_filename_list, query_filename_list, doc_list, query_list
pass

def get_dictionary(dataset):
    print('Creating dictionary')

    dictionary = []
    for doc in tqdm(dataset):
        dictionary += doc
    pass

    dictionary = list(dict.fromkeys(dictionary))

    return dictionary
pass

def compute_tf(dic, doc_list):
    print('computing tf...')

    tf = []

    with Pool(4) as p:
        for term in tqdm(dic):
            tf.append([doc.count(term) for doc in doc_list])
        pass
    pass

    return tf
pass

def compute_n(dic, doc_list):
    print('computing n...')

    n = []
    for term in tqdm(dic):

        count = 0
        for doc in doc_list:
            if term in doc:
                count += 1
        pass
        n.append(count)
    pass

    return n
pass

def compute_avg_len(l):
    if len(l) == 0:
        return 0 
    pass

    len_l = [len(e) for e in l]
    len_sum = sum(len_l)
    avg_len = len_sum / len(l)

    return avg_len
pass


def intersection(lst1, lst2):
   return [value for value in lst1 if value in lst2]
pass


def conbine_id_and_score(id_list, score_list):
    assert len(id_list) == len(score_list)

    conbined = [{'id': id_list[i], 'score': score_list[i]} for i in range(len(score_list))]
    return conbined
pass


def compute_score_for_querys(query_list):

    result = []
    for q, query in tqdm(enumerate(query_list)):
        score_list = compute_score(q, query)
        doc_score_list = conbine_id_and_list(doc_filename_list, score_list)
        doc_score_list.sort(key = lambda d: d['score'], reverse = True)
        sorted_doc_list = [doc['id'] for doc in doc_score_list]
        sorted_score_list = [doc['score'] for doc in doc_score_list]
        result.append({'id': query_filename_list[q], 'doc_list': sorted_doc_list})
    pass
    return result
pass

def write_result(path, result_list):
    with open(path, 'w') as out_file:
        out_file.write('Query,RetrievedDocuments\n')

        for result in result_list:
            out_file.write(result['id'] + ',')
            out_file.write(' '.join(result['doc_list']))
            out_file.write('\n')
        pass
    pass
pass

def compute_idf(N, n_list):
    DELTA = 0.5

    return [log((N - n + DELTA) / (n + DELTA)) for n in n_list]
pass

def init_prob(size):
    random_array = np.random.randint(100, size=size)
    summ = sum(random_array)
    return random_array / summ
pass

def _compute_e(args):
    p_tk_wi, p_dj_tk, step, j = args
    p___wi_tk = np.zeros([num_term, NUM_TOPIC])
    for i in range (num_term):
        for k in range(NUM_TOPIC):
            p___wi_tk[i, k] = p_tk_wi[k][i] * p_dj_tk[j][k]
        pass
    pass
pass

def compute_e(p_tk_wi, p_dj_tk, step):
    # p_dj_wi_tk = np.zeros([num_doc, num_term, NUM_TOPIC], dtype="float16")
    print("E step...")
    with Pool(cpu_count()) as p:
        args = [[p_tk_wi, p_dj_tk, step, j] for j in range(num_doc)]
        for _ in tqdm(p.imap_unordered(_compute_e, args), total=num_doc):
            pass
        pass
    pass
pass

doc_filename_list, query_filename_list, doc_list, query_list = get_data(data_dir)
dic = get_dictionary(doc_list)
num_doc = len(doc_list)
num_term = len(dic)
p_tk_wi = np.array([init_prob(num_term) for i in range(NUM_TOPIC)])
p_dj_tk = np.array([init_prob(NUM_TOPIC) for i in range(num_doc)])
print(cpu_count)
compute_e(p_tk_wi, p_dj_tk, 1)

#  avg_doc_len = compute_avg_len(doc_list)
#  tf_doc = compute_tf(dic, doc_list)
#  tf_query = compute_tf(dic, query_list)
#  n_list = compute_n(dic, doc_list)
#  idf_list = compute_idf(N, n_list)
#  result_list = compute_score_for_query(query_list)
#  write_result(OUT_PATH, result_list)
