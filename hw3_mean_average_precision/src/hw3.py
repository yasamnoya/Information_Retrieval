import random

def get_data():

    num_querys = int(input())

    pairs = []
    for i in range(num_querys):
        A_list = input().strip().split(' ')
        R_list = input().strip().split(' ')
        pairs.append({'A_list': A_list, 'R_list': R_list})
    pass

    return num_querys, pairs
pass

def MAP(num_querys, pairs):
    return sum([_MAP(pair) for pair in pairs]) / num_querys
pass

def _MAP(pair):

    A_list = pair['A_list']
    R_list = pair['R_list']
    is_relevant = [i for i, d in enumerate(A_list) if d in R_list]
    precisions = [(count + 1) / (i + 1) for count, i in enumerate(is_relevant)]
    avg_precision = sum(precisions) / len(R_list)
    
    return avg_precision
pass

num_querys, pairs = get_data()
if num_querys == 0:
    score = 1.0
else:
    score = MAP(num_querys, pairs)
pass

def my_round(num):

    num = str(num)
    if int(num[6]) >= 5:
        num_5 = str(int(num[5]) + 1)
    else:
        num_5 = num[5]
    pass

    return num[:5] + num_5 
pass    

print(round(score, 4), end = '')

