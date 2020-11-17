from tqdm import tqdm
from math import log
from multiprocessing import Pool
from sys import argv

import os



data_dir = 'ntust-ir-2020'
OUT_PATH = '../result.csv'



k1 = float(argv[1])
k3 = float(argv[2])
b = float(argv[3])

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

def BM25(k1, k3, b, len_dj, j, q, i_idx_list):

    acc = 0
    for i in i_idx_list:
        term1_child = (k1 + 1) * tf_doc[i][j]
        term1_mom = k1 * ((1 - b) + (b * (len_dj / avg_doc_len))) + tf_doc[i][j]
        term_1 = term1_child / term1_mom
        term_2 = ((k3 + 1) * tf_query[i][q]) / (k3 + tf_query[i][q])
        term_3 = idf_list[i]

        acc += term_1 * term_2 * term_3
    pass

    return acc
pass

#  [a, b, c, d, e] -> dic (123)
#  query ^ doc = [a, c]
#  i_idx_list = [0,2]

#  def BM25(k1, k3, b, len_dj, j, q, i_idx_list):
#
    #  acc = 0
    #  for i in i_idx_list:
        #  terms = []
        #  tf_ij_prime = tf_doc[i][j] / ((1 - b) + (b * (len_dj / avg_doc_len)))
        #  if tf_ij_prime > 0:
            #  term1_child = (k1 + 1) * (tf_ij_prime)
            #  term1_mom = k1 + tf_ij_prime
            #  terms.append(term1_child / term1_mom)
            #  terms.append((k3 + 1) * tf_query[i][q] / k3 + tf_query[i][q])
            #  terms.append(idf_list[i])
        #  else:
            #  continue
        #  pass
#
        #  acc += sum(terms)
    #  pass
#
    #  return acc
#  pass

#  def BM25L(k1, k3, b, len_dj, j, q, i_idx_list):
    #  DELTA = 0.1
#
    #  acc = 0
    #  for i in i_idx_list:
        #  terms = []
        #  tf_ij_prime = tf_doc[i][j] / ((1 - b) + (b * (len_dj / avg_doc_len)))
        #  if tf_ij_prime > 0:
            #  term1_child = (k1 + 1) * (tf_ij_prime + DELTA)
            #  term1_mom = k1 + tf_ij_prime + DELTA
            #  terms.append(term1_child / term1_mom)
            #  terms.append(((k3 + 1) * tf_query[i][q]) / (k3 + tf_query[i][q]))
            #  terms.append(idf_list[i])
        #  else:
            #  continue
        #  pass
#
        #  acc += sum(terms)
    #  pass
#
    #  return acc
#  pass

def intersection(lst1, lst2):
   return [value for value in lst1 if value in lst2]
pass

def _compute_score(args):
    q, query, j, doc = args
    intersection_q_doc = intersection(query, doc)
    i_idx_list = [i for i, term in enumerate(dic) if term in intersection_q_doc]
    score = BM25(k1, k3, b, len(doc), j, q, i_idx_list)
    return score
pass

def compute_score(q, query):

    #  score_list = []
    #  for j, doc in enumerate(doc_list):
        #  i_idx_list = [i for i in range(len(dic)) if dic[i] in intersection(query, doc)]
        #  score = BM25(k1, k3, b, len(doc), j, q, i_idx_list)
        #  score_list.append(score)
    #  pass

    args = [[q, query, j, doc] for j, doc in enumerate(doc_list)]
    with Pool(4) as p:
        score_list = p.map(_compute_score, args)
    pass

    return score_list
pass

def conbine_id_and_list(id_list, l):
    assert len(id_list) == len(l)

    conbined = [{'id': id_list[i], 'score': l[i]} for i in range(len(l))]
    return conbined
pass


def compute_score_for_query(query_list):

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

doc_filename_list, query_filename_list, doc_list, query_list = get_data(data_dir)
N = len(doc_list)
avg_doc_len = compute_avg_len(doc_list)
dic = get_dictionary(query_list)
tf_doc = compute_tf(dic, doc_list)
tf_query = compute_tf(dic, query_list)
n_list = compute_n(dic, doc_list)
idf_list = compute_idf(N, n_list)
result_list = compute_score_for_query(query_list)
write_result(OUT_PATH, result_list)
