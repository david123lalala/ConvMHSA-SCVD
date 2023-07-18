import os
import pandas
from clean_fragment import clean_fragment
from vectorize_fragment import FragmentVectorizer
from parser import parameter_parser
import json

from models.ConvMHSA import ConvMHSA_Model

import numpy as np
import pickle
import argparse

args = parameter_parser()

reentrancy_DD_path='data/extract_reentrancy_function_single/'
reentrancy_DDCD_path='data/extract_function_reentrancy/'
reentrancy_none_path='data/reentrancy_source_code_pkl/'

file_reentrancy_name_path='data/re_final_reentrancy_name.txt'
file_reentrancy_value_path='data/re_final_reentrancy_label.txt'

for arg in vars(args):
    print(arg, getattr(args, arg))

def get_fragment(tmp):
    fragment=[]
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            fragment.append(tmp[i][j])
    return fragment



def get_vectors_df(filename, vector_length=300):
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    file_num=[]
    fragments_str=[]

    if filename=='reentrancy_DD':
        file_path=reentrancy_DD_path
        file_name=np.loadtxt(file_reentrancy_name_path,delimiter=',')
        file_value=np.loadtxt(file_reentrancy_value_path,delimiter=',')
    elif filename=='reentrancy_DDCD':
        file_path=reentrancy_DDCD_path
        file_name=np.loadtxt(file_reentrancy_name_path,delimiter=',')
        file_value=np.loadtxt(file_reentrancy_value_path,delimiter=',')
    elif filename=='reentrancy_none':
        file_path=reentrancy_none_path
        file_name=np.loadtxt(file_reentrancy_name_path,delimiter=',')
        file_value=np.loadtxt(file_reentrancy_value_path,delimiter=',')

    for i in range(len(file_name)):
        tmp_file_path=file_path+str(int(file_name[i]))+'.pkl'

        with open(tmp_file_path,'rb') as f:
            tmp=pickle.load(f)
            fragment=get_fragment(tmp)

        print("Collecting code_gadgets...",i, end="\r")
        vectorizer.add_fragment(fragment)
        val=int(file_value[i])
        row={'fragment':fragment,'val':val}
        fragments.append(row)
    print( )
    print("Training model...", end="\r")
    vectorizer.train_model()
    vectors = []
    count = 0

    for fragment in fragments:
        count += 1
        print("Processing code_gadgets...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)

    df = pandas.DataFrame(vectors)
    return df

def main():
    filename = args.dataset
    vector_file_path = filename + "_code_gadget_vectors.pkl"
    vector_length = args.vector_dim

    if os.path.exists(vector_file_path):
        # df=pandas.DataFrame(file)
        df = pandas.read_pickle(vector_file_path)
        # print(df[0])
        # print(df[1])
    else:
        df = get_vectors_df(filename, vector_length)
        df.to_pickle(vector_file_path)

    if args.model == 'ConvMHSA':
        model =ConvMHSA_Model(df,filename)
    model.train()
    model.test()

if __name__ == "__main__":
    main()
