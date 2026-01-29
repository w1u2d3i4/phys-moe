#构建C × C 的规则相似度矩阵 S_rule

import os
import pandas as pd
import string
import time
from tqdm import tqdm

#处理space_groups_data_cleaned.csv文件，构建规则相似度矩阵(用余弦相似度)
def build_rule_similarity_matrix(input_csv, unique_file):
    #读取space_groups_data_cleaned.csv文件
    df = pd.read_csv(input_csv)
    data = df['symmetry_operators_cleaned'].tolist()
    for i in range(len(data)):
        data[i] = data[i].replace(" ", "").replace('"', "")
        operators = data[i].strip("['']").split("','")
        data[i] = operators
    with open(unique_file, 'r') as f:
        unique_operators = f.read().splitlines()
    unique_dict = {op: idx for idx, op in enumerate(unique_operators)}
    sg_vector = []
    for i in range(len(data)):
        vector = [0] * len(unique_operators)
        for op in data[i]:
            if op in unique_dict:
                vector[unique_dict[op]] += 1
        sg_vector.append(vector)
    #计算规则相似度矩阵
    C = len(sg_vector)
    S_rule = [[0.0 for _ in range(C)] for _ in range(C)]
    for i in tqdm(range(C), desc="Building rule similarity matrix"):
        vec_i = sg_vector[i]
        norm_i = sum(x * x for x in vec_i) ** 0.5
        for j in range(i, C):
            vec_j = sg_vector[j]
            norm_j = sum(x * x for x in vec_j) ** 0.5
            dot_product = sum(vec_i[k] * vec_j[k] for k in range(len(unique_operators)))
            if norm_i > 0 and norm_j > 0:
                similarity = dot_product / (norm_i * norm_j)
            else:
                similarity = 0.0
            S_rule[i][j] = similarity
            S_rule[j][i] = similarity
    #保存规则相似度矩阵
    output_file = 'rule_matrix.csv'
    S_rule_df = pd.DataFrame(S_rule)
    S_rule_df.to_csv(output_file, index=False, header=False)
    print(f"Rule similarity matrix saved to {output_file}")

def read_rule_similarity_matrix(matrix_file):
    S_rule_df = pd.read_csv(matrix_file, header=None)
    S_rule = S_rule_df.values.tolist()
    return S_rule

def set_sg_to_sys_map():
    #设置从晶系到空间群的矩阵
    #设置成230个元素的list，取值在0~6之间，表示每个空间群对应的晶系类别
    #1-2：三斜晶系；3-15：单斜晶系；16-74：正交晶系；75-142：四方晶系；
    #143-167：三方晶系；168-194：六方晶系；195-230：立方晶系
    sg_to_sys = 2 * [0] + 13 * [1] + 59 * [2] + 68 * [3] + 25 * [4] + 27 * [5] + 36 * [6]
    return sg_to_sys


if __name__ == "__main__":
    input_csv = '/opt/data/private/ICL/space_groups_data_cleaned.csv'
    unique_file = '/opt/data/private/ICL/unique_operators.txt'
    build_rule_similarity_matrix(input_csv, unique_file)
    S_rule = read_rule_similarity_matrix('/opt/data/private/ICL/rule_matrix.csv')
    print(S_rule[:5])

    

    