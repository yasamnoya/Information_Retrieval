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


NUM_TOPIC = 8
DTYPE = nb.float32

alpha = float(argv[1])
beta = float(argv[2])

STEP = int(argv[3])
EPOCH = int(argv[4])

evaluate = int(argv[5])


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

def get_most_count_mask(dic, c, th):
    mask = []
    for i in reversed(range(len(dic))):
        c_sum = np.sum(np.array(c[i]))
        mask.append(c_sum >= th)
    pass
    return mask
pass

def get_term_to_i(dic):

    term_to_i = {}
    for i, term in enumerate(dic):
        term_to_i[term] = i
    pass
    return term_to_i
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

def _compute_score(args):
    query_i_list, j = args
    summ = 0
    for i in query_i_list:
        term1 = np.log(alpha) + np.log(p_dj_wi[j, i])
        term2 = np.log(beta) + np.log(sum([p_tk_wi[k, i] * p_dj_tk[j, k] for k in range(NUM_TOPIC)]))
        term3 = np.log(1-alpha-beta) + np.log(p_bg_wi[i])
        summ += np.logaddexp(term1, np.logaddexp(term2, term3))
    pass
    return summ
pass

def compute_score(q, query):
    query_i_list = [term_to_i[term] for term in query if term in dic]
    with Pool(cpu_count()) as p:
        args = [[query_i_list, j] for j in range(num_doc)]
        score_list = p.map(_compute_score, args)
    pass
    return score_list
pass


def compute_score_for_querys(query_list):

    result = []
    for q, query in tqdm(enumerate(query_list)):
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

def compute_idf(N, n_list):
    DELTA = 0.5

    return [log((N - n + DELTA) / (n + DELTA)) for n in n_list]
pass

def init_prob(size):
    random_array = np.random.randint(1, 10, size=size)
    summ = sum(random_array)
    return random_array / summ
pass

def compute_c():
    global c_wi_dj
    print("Counting c...", datetime.now().strftime("%H:%M:%S"))
    with Pool(cpu_count()) as p:
        args = [i for i in range(num_term)]
        #  for _ in tqdm(p.imap_unordered(_compute_c, args), total=len(args)):
            #  pass
        #  pass
        c_wi_dj = np.ndarray(p.map(_compute_c, args))
    pass
    #  np.save("../save/c", c_wi_dj)
    return c_wi_dj
pass

def _compute_c(i):
    term = dic[i]
    for j in range(num_doc):
        c_wi_dj[i, j] = doc_list[j].count(term)
    pass
    return c_wi_dj[i].tolist()
pass

# @nb.jit(nopython=True)
# def _compute_e(j):
#     p___wi_tk = np.zeros((num_term, NUM_TOPIC), dtype=nb.float32)
#     for i in range (num_term):
#         for k in range(NUM_TOPIC):
#             p___wi_tk[i, k] = p_tk_wi[k][i] * p_dj_tk[j][k]
#             #  p_dj_wi_tk[j, i] = normalize(p_dj_wi_tk[j, i], 0)
#             #  p_dj_wi_tk[j, i, k] = p_tk_wi[k][i] * p_dj_tk[j][k]
# 
#         pass
#         p___wi_tk[i] /= np.sum(p___wi_tk[i])
#         #  p_dj_wi_tk[j, i] /= sum(p_dj_wi_tk[j, i])
#     pass
#     return p___wi_tk
# pass

@nb.njit(parallel=True)
def _compute_e():
    p_dj_wi_tk = np.zeros((num_doc, num_term, NUM_TOPIC), dtype=DTYPE)
    for j in range(num_doc):
        for i in range(num_term):
            for k in range(NUM_TOPIC):
                p_dj_wi_tk[j, i, k] = p_tk_wi[k][i] * p_dj_tk[j][k]
            pass
        p_dj_wi_tk[j, i] /= np.sum(p_dj_wi_tk[j, i])
        pass
    pass
    return p_dj_wi_tk
pass

def compute_e(step):
    print("E step%d..." %step, datetime.now().strftime("%H:%M:%S"))
    #  args = [j for j in range(num_doc)]
    #  for _ in tqdm(p.imap_unordered(_compute_e, args), total=len(args)):
        #  pass
    #  pass
     
    #  p_dj_wi_tk = np.array(p.map(_compute_e, args))
    #  print("To sparse...")
    #  p_dj_wi_tk = sp.csr_matrix(p_dj_wi_tk)
    #  print("Stacking...")
    #  p_dj_wi_tk = np.vstack(p_dj_wi_tk)

    #  p_dj_wi_tk = []
    #  for j in tqdm(args):
        #  p___wi_tk = sp.csr_matrix(_compute_e(j))
        #  p_dj_wi_tk.append(p___wi_tk)
        #  sp.save_npz('../save/e_step%d_%d.npz' %(step, j), p___wi_tk)
    #  pass

    #  print("Saving...")
    #  for i, m in tqdm(enumerate(p_dj_wi_tk)):
        #  sp.save_npz('../save/e_step%d_%d.npz' %(step, i), m)
    #  pass
    #  sp.save_npz('../save/e_step%d.npz' %step, p_dj_wi_tk)

    p_dj_wi_tk = _compute_e()
    return p_dj_wi_tk
pass

#  def _compute_m_1(k):
    #  for i in range(num_term):
        #  p_tk_wi[k, i] = sum([c_wi_dj[i, j] * p_dj_wi_tk[j, i, k] for j in range(num_doc) if c_wi_dj[i, j] > 0])
    #  pass
    #  p_tk_wi[k] /= sum(p_tk_wi[k])
    #  return p_tk_wi[k].tolist()
#  pass

@nb.njit(parallel=True)
def _compute_m_1():
    p_tk_wi = np.zeros((NUM_TOPIC, num_term), dtype=DTYPE)
    for k in range(NUM_TOPIC):
        for i in range(num_term):
            p_tk_wi[k, i] = np.sum(np.array([c_wi_dj[i, j] * p_dj_wi_tk[j, i, k] for j in range(num_doc) if c_wi_dj[i, j] > 0]))
            #  p_tk_wi[k, i] = p_dj_wi_tk[0, i, k]
            #  a = p_dj_wi_tk[0, i, k]
        pass
        p_tk_wi[k] /= np.sum(p_tk_wi[k])
    pass
    return p_tk_wi
pass

def compute_m_1(step):
    print("M_1 step%d..." %step, datetime.now().strftime("%H:%M:%S"))
    #  global p_tk_wi
    #  p_dj_wi_tk = np.load("../save/e_step1.npy")
    #  c_wi_dj = np.load("../save/c.npy")
    # with Pool(cpu_count()) as p:
    #     args = [k for k in range(NUM_TOPIC)]
    #     #  for _ in tqdm(p.imap_unordered(_compute_m_1, args), total=len(args)):
    #         #  pass
    #     #  pass
    #     p_tk_wi = np.array(p.map(_compute_m_1, args))
    # pass
    #  np.save("../save/m_1_step%d" %step, p_tk_wi)
    p_tk_wi = _compute_m_1()
    return p_tk_wi
pass

#  def _compute_m_2(j):
    #  len_dj = len(doc_list[j])
    #  for k in range(NUM_TOPIC):
        #  p_dj_tk[j, k] = sum([c_wi_dj[i, j] * p_dj_wi_tk[j, i, k] for i in range(num_term) if c_wi_dj[i, j] > 0]) / len_dj
    #  pass
    #  return p_dj_tk[j].tolist()
#  pass

@nb.njit(parallel=True)
def _compute_m_2():
    p_dj_tk = np.zeros((num_doc, NUM_TOPIC), dtype=DTYPE)
    for j in range(num_doc):
        doc_len = len_dj[j]
        for k in range(NUM_TOPIC):
            p_dj_tk[j, k] = np.sum(np.array([c_wi_dj[i, j] * p_dj_wi_tk[j, i, k] for i in range(num_term) if c_wi_dj[i, j] > 0]))
        pass
        p_dj_tk[j] /= doc_len
    pass
    return p_dj_tk
pass

def compute_m_2(step):
    print("M_2 step%d..." %step, datetime.now().strftime("%H:%M:%S"))
    #  global p_dj_tk
    #  p_dj_wi_tk = np.load("../save/e_step1.npy")
    #  c_wi_dj = np.load("../save/c.npy")
    # with Pool(cpu_count()) as p:
    #     args = [j for j in range(num_doc)]
    #     #  for _ in tqdm(p.imap_unordered(_compute_m_2, args), total=len(args)):
    #         #  pass
    #     #  pass
    #     p_dj_tk = np.array(p.map(compute_m_2, args))
    # pass
    # np.save("../save/m_2_step%d" %step, p_dj_tk)
    p_dj_tk = _compute_m_2()
    return p_dj_tk
pass

def compute_p_bg_wi(c_wi_dj, doc_list):
    mom = np.sum(np.array([len(doc) for doc in doc_list]))
    return _compute_p_bg_wi(c_wi_dj, mom)
pass

@nb.njit(parallel=True)
def _compute_p_bg_wi(c_wi_dj, mom):
    #  print("Building BG...", datetime.now().strftime("%H:%M:%S"))
    #  c_wi_dj = np.load("../save/c.npy")
    p_bg_wi = np.zeros(num_term, dtype=DTYPE)
    #  mom = 0
    #  for doc in doc_list:
        #  mom += len(doc)
    #  pass
    #  mom = np.sum(np.array([len(doc) for doc in doc_list]))
    for i in range(num_term):
        child = np.sum(c_wi_dj[i])
        p_bg_wi[i] = child / mom
    pass
    #  np.save("../save/p_bg_wi", p_bg_wi)
    return p_bg_wi
pass

def compute_p_dj_wi(c_wi_dj, doc_list):
    doc_len_list = np.array([len(doc) for doc in doc_list])
    return _compute_p_dj_wi(c_wi_dj, doc_len_list)
pass

@nb.njit(parallel=True)
def _compute_p_dj_wi(c_wi_dj, doc_len_list):
    #  print("Building unigram...", datetime.now().strftime("%H:%M:%S"))
    #  c_wi_dj = np.load("../save/c.npy")
    p_dj_wi = np.zeros((num_doc, num_term), dtype=DTYPE)
    for j in range(num_doc):
        for i in range(num_term):
            p_dj_wi[j, i] = c_wi_dj[i, j] / doc_len_list[j]
        pass
    pass
    #  np.save("../save/p_dj_wi", p_dj_wi)
    return p_dj_wi
pass

def normalize(m, axis):
    summ = m.sum(axis=axis, keepdims=True)
    return  m / summ
pass


###############test#############
#  a = np.arange(0,27,1).reshape(3,3,3)num_term
#  print(a[1,:,1])
#  a[2] = a[1]
#  print(a)
#  print(a)
#  print(normalize(a, 2))

#  p_dj_wi_tk = np.load("../save/e_step1.npy")
#  normalize(p_dj_wi_tk, 2)
#  print(p_dj_wi_tk.shape)



#  exit()
################################
print("Start...", datetime.now().strftime("%H:%M:%S"))

doc_filename_list, query_filename_list, doc_list, query_list = get_data(data_dir)
dic = get_dictionary(doc_list)

#  c init
try:
    print("Loading c_wi_dj...")
    c_wi_dj = sp.load_npz("../save/s_c.npz").todense()
except:
    c_wi_dj = np.zeros([num_term, num_doc])
    compute_c()

mask = get_most_count_mask(dic, c_wi_dj,  1)
c_wi_dj = np.vstack([c_wi_dj[i] for i in range(len(dic)) if mask[i]])
dic = [dic[i] for i in range(len(dic)) if mask[i]]
print(len(dic))
term_to_i = get_term_to_i(dic)
num_doc = len(doc_list)
num_term = len(dic)
len_dj = np.array([len(doc_list[j]) for j in range(num_doc)])

# computing LMs
try:
    raise 'skip'
    print("Loading p_bg_wi...")
    p_bg_wi = sp.load_npz("../save/s_p_bg_wi.npz")
    print("Loading p_dj_wi...")
    p_dj_wi = sp.load_npz("../save/s_p_dj_wi.npz")
except:
    p_bg_wi = compute_p_bg_wi(c_wi_dj, doc_list)
    p_dj_wi = compute_p_dj_wi(c_wi_dj, doc_list)



# init
if STEP == 0:
    p_tk_wi = np.array([init_prob(num_term) for i in range(NUM_TOPIC)], dtype=np.float32)
    p_dj_tk = np.array([init_prob(NUM_TOPIC) for i in range(num_doc)], dtype=np.float32)

    #  p_dj_wi_tk = np.zeros([num_doc, num_term, NUM_TOPIC], dtype="float16")

#load
else:
    p_tk_wi = sp.load_npz("../save/step %d_1.npz" %STEP)
    p_dj_tk = sp.load_npz("../save/step %d_2.npz" %STEP)
    p_tk_wi = p_tk_wi.todense()
    p_dj_tk = p_dj_tk.todense()

#  p_dj_wi_tk = np.load("../save/e_step1.npy")

#  training
p_dj_wi_tk = 0
for step in range(STEP + 1,STEP + 1 + EPOCH):

    p_dj_wi_tk = 0
    p_dj_wi_tk = compute_e(step)


    p_tk_wi = 0
    p_dj_tk = 0
    p_tk_wi = compute_m_1(step)
    sp1 = sp.csr_matrix(p_tk_wi)
    print("Saving... ")
    sp.save_npz("../save/step %d_1" %step, sp1)

    p_dj_tk = compute_m_2(step)
    sp2 = sp.csr_matrix(p_dj_tk)
    print("Saving... ")
    sp.save_npz("../save/step %d_2" %step, sp2)

pass


#=============================================
#  avg_doc_len = compute_avg_len(doc_list)
#  tf_doc = compute_tf(dic, doc_list)
#  tf_query = compute_tf(dic, query_list)
#  n_list = compute_n(dic, doc_list)
#  idf_list = compute_idf(N, n_list)

p_dj_wi_tk = 0
#  p_bg_wi = p_bg_wi.todense().tolist()[0]
#  d_dj_wi = p_dj_wi.todense()
print(p_tk_wi.shape)
print(p_dj_tk.shape)
print(len(p_bg_wi))
print(p_dj_wi.shape)

if evaluate > 0:
    result_list = compute_score_for_querys(query_list)
    write_result(OUT_PATH, result_list)
