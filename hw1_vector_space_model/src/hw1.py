from tqdm import tqdm
from math import log, sqrt


data_dir = '../ntust-ir-2020'

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
        for word in doc:
            dictionary.append(word)
        pass
    pass

    dictionary = list(dict.fromkeys(dictionary))

    return dictionary
pass

def compute_tf(dic, doc_list):
    print('computing tf...')

    tf = []
    for term in tqdm(dic):
        tf.append([doc.count(term) / len(doc) for doc in doc_list])
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

def compute_d(tf, N, n):
    #  return tf + log(N / (n + 1e-1))
    return (1 + tf) * log(N / (n + 1e-1))
    #  return (1 + log(1 + tf))
pass

def compute_q(tf, N, n, max_tf):
    #  return(0.5 + 0.5 * (tf / max_tf)) * log(N / (n + 1e-1))
    return (1 + tf) * log(N / (n + 1e-1))
    #  return log(N / (n + 1e-1) + 1)
pass

def length(l):
    return sqrt(sum(map(lambda x: x**2, l)))
pass

def cos_sim(a, b):
    assert len(a) == len(b)
    a_dot_b = sum([a[i] * b[i] for i in range(len(a))])

    return a_dot_b / (length(a) * length(b))
pass
    
doc_filename_list, query_filename_list, doc_list, query_list = get_data(data_dir)
dic = get_dictionary(query_list)
tf_doc = compute_tf(dic, doc_list)
tf_query = compute_tf(dic, query_list)
n_list = compute_n(dic, doc_list)

with open('../result.csv', 'w') as out_file:
    out_file.write('Query,RetrievedDocuments\n')
    
    print('computing score...')
    for k, query in tqdm(enumerate(query_list)):
        
        #  q_appear_idx = [i for i in range(len(dic)) if dic[i] in query]
        q_appear_idx = range(len(dic))

        max_tf = max([tf_query[i][k] for i in q_appear_idx])
        q_vector = [compute_q(tf_query[i][k], len(doc_list),  n_list[i], max_tf) for i in q_appear_idx]

        d_vector_list = []
        for j, doc in enumerate(doc_list):
            d_vector = [compute_d(tf_doc[i][j], len(doc_list), n_list[i]) for i in q_appear_idx]
            d_vector_list.append(d_vector)
        pass
        
        score_list = [cos_sim(q_vector, d_vector) for d_vector in d_vector_list]
        filename_score_pair_list = [[doc_filename_list[i], score] for i, score in enumerate(score_list)]
        
        filename_score_pair_list.sort(key = lambda l: l[1], reverse=True)

        
        out_file.write(query_filename_list[k] + ',')

        to_write_filenames = ' '.join([filename_score_pair_list[i][0] for i in range(len(filename_score_pair_list))])
        out_file.write(to_write_filenames)
        out_file.write('\n')
    pass
pass

