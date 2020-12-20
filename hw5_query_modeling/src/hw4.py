from tqdm import tqdm
from math import log
from multiprocessing import Pool, cpu_count
from sys import argv
from datetime import datetime
from scipy import sparse as sp

import numpy as np
import numba as nb
import os
import gc
import h5py


NUM_TOPIC = 32
DTYPE = nb.float32

alpha = float(argv[1])
beta = float(argv[2])

STEP = int(argv[3])
EPOCH = int(argv[4])
TH = int(argv[5]) 



data_dir = '../ntust-ir-2020_hw4_v2'
OUT_PATH = '../result.csv'


def save_h5(filename, data, compression="lzf"):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(filename, data=data)
pass

def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        data = f[filename][:]
    return data
pass

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

@nb.njit
def _compute_idf(doc_in_id, len_doc):
    part_of_idf_wi = np.zeros(num_term, dtype=nb.int32)

    for k in range(len_doc):
        part_of_idf_wi[doc_in_id[k]] = 1
    pass

    return part_of_idf_wi
pass

def compute_idf():
    print("Computing idf...")

    idf_wi = np.zeros(num_term, dtype=np.int32)
    for j in tqdm(range(num_doc)):
        idf_wi += _compute_idf(doc_list_in_id[j], len(doc_list_in_id[j]))
    pass

    return idf_wi
pass

def sort_by_idf(dic, idf_wi):
    conbined = [{'term': dic[i], 'idf': idf_wi[i]} for i in range(num_term)]
    conbined.sort(key = lambda t: t['idf'], reverse=True)
    sorted_dic = [conbined[i]['term'] for i in range(num_term)]
    idf_wi = [conbined[i]['idf'] for i in range(num_term)]

    return sorted_dic, idf_wi
pass

def get_term_to_i(dic):

    term_to_i = {}
    for i, term in enumerate(dic):
        term_to_i[term] = int(i)
    pass
    return term_to_i
pass

def truncate_dic(dic):
    for i in range(num_term):
        if idf_wi[i] < TH:
            return dic[:i]
        pass
    pass
    return dic
pass

def conbine_id_and_score(id_list, score_list):
    assert len(id_list) == len(score_list)

    conbined = [{'id': id_list[i], 'score': score_list[i]} for i in range(len(score_list))]
    return conbined
pass

@nb.njit
def _compute_score(query_in_id, len_q):
    print(np.sum(p_tk_wi, axis=1))
    print(np.sum(p_tk_wi, axis=1))
    print("-")
    score_list = np.zeros(num_doc, nb.float32)
    for j in range(num_doc):
        for i in range(len_q):

            if i < num_term:
                term1 = np.log(alpha) + np.log(p_dj_wi[j, query_in_id[i]])
                term2 = np.log(beta) + np.log(np.sum(p_tk_wi[:, query_in_id[i]] * p_tk_dj[:, j]))
                #  print(p_tk_wi[:, query_in_id[i]] ,  p_tk_dj[:, j])
                #  term2 = np.log(0)
                term3 = np.log(1 - alpha-beta) + np.log(p_bg_wi[query_in_id[i]])
                score_list[j] += np.logaddexp(term1, np.logaddexp(term2, term3))
            #  else:
                #  term = 0
            pass

        pass
    pass
    return score_list
pass

def compute_score(q, query):

    query_in_id = np.array([term_to_i[term] for term in query if term in dic_full], dtype=np.int32)

    if(len(query_in_id) > 0):
        score_list = _compute_score(query_in_id, query_in_id.shape[0])
    else:
        score_list = np.array([0] * num_doc)

    return score_list
pass


def compute_score_for_querys(query_list):
    print("Computing scores...")

    result = []
    for q, query in tqdm(enumerate(query_list), total=len(query_list)):
        score_list = compute_score(q, query)
        doc_score_list = conbine_id_and_score(doc_filename_list, score_list)
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

def doc_to_id(doc_list):
    print("Converting docs into id...")
    return np.array([np.array([term_to_i[term] for term in doc]) for doc in tqdm(doc_list)])
pass

@nb.njit
def _compute_c(args):
    doc_in_id, doc_len = args

    c___wi = np.zeros(num_term, dtype=nb.int16)

    for t in range(doc_len):
        c___wi[doc_in_id[t]] += 1
    pass

    return c___wi
pass

def compute_c(doc_list_in_id):
    print("Computing C...")
    
    args = [(np.array(doc_in_id), len(doc_in_id)) for doc_in_id in doc_list_in_id]
    c_dj_wi = np.array([_compute_c(arg) for arg in tqdm(args)])

    return c_dj_wi
pass

@nb.njit
def _compute_bg(i, all_doc_len):
    return np.sum(c_dj_wi[:,i]) / all_doc_len
pass

def compute_bg(c_dj_wi, doc_list):
    print("Computing BG LM...")

    all_doc_len = np.sum([len(doc) for doc in doc_list])

    c_bg_wi = np.array([_compute_bg(i, all_doc_len) for i in tqdm(range(num_term))])
    return c_bg_wi
pass

def compute_ULM(c_dj_wi, doc_list):
    print("Computing ULM...")

    p_dj_wi = np.array([c_dj_wi[j,:] / len(doc_list[j]) for j in tqdm(range(num_doc))])
    return p_dj_wi
pass

@nb.njit 
def init_p_tk_wi():
    p_tk_wi = np.random.rand(NUM_TOPIC, num_term)

    for k in range(NUM_TOPIC):
        p_tk_wi[k] /= np.sum(p_tk_wi[k])
    pass
    return p_tk_wi
pass

@nb.njit 
def init_p_tk_dj():
    p_tk_dj = np.random.rand(NUM_TOPIC, num_doc)

    for k in range(NUM_TOPIC):
        p_tk_dj[k] /= np.sum(p_tk_dj[k])
    pass
    return p_tk_dj
pass

@nb.njit
def _e_step(k, p_tk_wi, p_tk_dj):
    p___dj_wi = np.zeros((num_doc, num_term), dtype=DTYPE)

    for j in range(num_doc):
        for i in range(num_term):
            p___dj_wi[j, i] = p_tk_wi[k, i] * p_tk_dj[k ,j]
        pass
    pass

    return p___dj_wi
pass

@nb.njit
def _m_1_step(k, p___dj_wi):
    p___wi = np.zeros(num_term, dtype=DTYPE)

    for i in range(num_term):
        for j in range(num_doc):
            p___wi[i] += c_dj_wi[j, i] * p___dj_wi[j, i]
        pass
    pass

    p___wi /= np.sum(p___wi)
    return p___wi
pass

@nb.njit
def _m_2_step(k, p___dj_wi, len_dj):
    p___dj = np.zeros(num_doc, dtype=DTYPE)

    for j in range(num_doc):
        for i in range(num_term):
            p___dj[j] += c_dj_wi[j, i] * p___dj_wi[j, i]
        pass
        p___dj /= len_dj[j]
    pass

    return p___dj
pass

def train_epoch(step, p_tk_wi, p_tk_dj):
    print("E step %d..." %step)

    sum_p___dj_wi = np.zeros((num_doc, num_term), dtype=np.float32)

    for k in tqdm(range(NUM_TOPIC)):
        p___dj_wi = _e_step(k, p_tk_wi, p_tk_dj)
        sum_p___dj_wi = sum_p___dj_wi + p___dj_wi
        save_h5("../save/%d_topics.e_step_%d_.%d.h5" %(NUM_TOPIC, step, k), p___dj_wi)
    pass


    p_tk_wi = np.zeros((NUM_TOPIC, num_term), dtype=np.float32)
    p_tk_dj = np.zeros((NUM_TOPIC, num_doc), dtype=np.float32)


    print("M step %d..." %step)

    len_dj = np.array([len(doc) for doc in doc_list], dtype=np.int32)
    for k in tqdm(range(NUM_TOPIC)):
        p___dj_wi = load_h5("../save/%d_topics.e_step_%d_.%d.h5" %(NUM_TOPIC, step, k))

        p___dj_wi = np.nan_to_num(p___dj_wi / sum_p___dj_wi)
        p_tk_wi[k] = np.nan_to_num(_m_1_step(k, p___dj_wi))
        p_tk_dj[k] = np.nan_to_num(_m_2_step(k, p___dj_wi, len_dj))

        save_h5("../save/%d_topics.e_step_%d_.%d.h5" %(NUM_TOPIC, step, k), p___dj_wi)
    pass

    #  save_h5("../save/%d_topics.m_1_step_%d.h5" %(NUM_TOPIC, step), p_tk_wi)
    #  save_h5("../save/%d_topics.m_2_step_%d.h5" %(NUM_TOPIC, step), p_tk_dj)

    print(p_tk_wi)
    print(p_tk_dj)

    return p_tk_wi, p_tk_dj
pass


###############test#############

#  exit()
################################
print("Start...", datetime.now().strftime("%H:%M:%S"))

doc_filename_list, query_filename_list, doc_list, query_list = get_data(data_dir)
num_doc = len(doc_list)
dic = get_dictionary(doc_list)
num_term = len(dic)
term_to_i = get_term_to_i(dic)
doc_list_in_id = np.array(doc_to_id(doc_list))
idf_wi = compute_idf()

dic, idf_wi = sort_by_idf(dic, idf_wi)
term_to_i = get_term_to_i(dic)
doc_list_in_id = np.array(doc_to_id(doc_list))

c_dj_wi = compute_c(doc_list_in_id)
p_bg_wi = compute_bg(c_dj_wi, doc_list)
p_dj_wi = compute_ULM(c_dj_wi, doc_list)

dic_full = dic
dic = truncate_dic(dic)
num_term = len(dic)
print(len(dic))
input("..................")


## init
p_tk_wi = np.zeros((NUM_TOPIC, num_term), dtype=np.float32)
p_tk_dj = np.zeros((NUM_TOPIC, num_doc), dtype=np.float32)

if STEP == 0:
    p_tk_wi = np.array(init_p_tk_wi())
    p_tk_dj = np.array(init_p_tk_dj())
else:
    p_tk_wi = load_h5("../save/%d_topics.m_1_step_%d.h5" %(NUM_TOPIC, STEP))
    p_tk_dj = load_h5("../save/%d_topics.m_2_step_%d.h5" %(NUM_TOPIC, STEP))
pass

for step in range(STEP + 1, STEP + EPOCH + 1):

    p_tk_wi, p_tk_dj = train_epoch(step, p_tk_wi, p_tk_dj)

    if step % 1 == 0:
        print("Saving to ../save/%d_topics.m_1_step_step_%d.h5" %(NUM_TOPIC, step))
        save_h5("../save/%d_topics.m_1_step_%d.h5" %(NUM_TOPIC, step), p_tk_wi)
        print("Saving to ../save/%d_topics.m_2_step_step_%d.h5" %(NUM_TOPIC, step))
        save_h5("../save/%d_topics.m_2_step_%d.h5" %(NUM_TOPIC, step), p_tk_dj)

    print()
    print(p_tk_wi)
    print(p_tk_dj)
    print("----")

    pass
pass


result_list = compute_score_for_querys(query_list)
write_result(OUT_PATH, result_list)
