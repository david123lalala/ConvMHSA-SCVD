import re
import warnings
import numpy as np
from gensim.models import Word2Vec
import sys

warnings.filterwarnings("ignore")

target_line='call.value'
target_line2='block.timestamp'
target_line3='block.number'
target_line4='for'
target_line4='while'


# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}

"""
Functionality to train Word2Vec model and vectorize fragments
Trains Word2Vec model using list of tokenized fragments
Uses trained model embeddings to create 2D fragment vectors
"""


class FragmentVectorizer:
    def __init__(self, vector_length):
        self.fragments = []
        self.vector_length = vector_length
        #forward_slices的意思
        #backward——slices的意思
        self.forward_slices = 0
        self.backward_slices = 0

    """
    Takes a line of solidity code (string) as input
    Tokenizes solidity code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """
    #输入solidity的一行代码
    #tmp用来保存solidity的一段序列，w保存一个完整的单词
    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        
        # if re.search(target_line2,line,flags=0):
            # tmp=[]
        # elif re.search(target_line3,line,flags=0):
            # tmp=[]
        # else:
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                    #''.join表示生成新的字符串
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
                    # Check operators and append to final list
            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 3])
                w = []
                i += 3
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
                # Character appended to word list
            else:
                w.append(line[i])
                i += 1
                # Filter out irrelevant strings

        res = list(filter(lambda c: c != '', tmp))
        # res = list(filter(lambda c: c != 'call', res))
        # res = list(filter(lambda c: c != 'value', res))
        return list(filter(lambda c: c != ' ', res))


    """
    Tokenize entire fragment
    Tokenize each line and concatenate to one long list
    """

    #将fragment转变成tokenize
    @staticmethod
    def tokenize_fragment(fragment):
        tokenized = []
        #re.compile 正则表达式生成对象
        function_regex = re.compile('function(\d)+')
        backwards_slice = False
        for line in fragment:
            #向量化line
            tokens = FragmentVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    """
    Add input fragment to model
    Tokenize fragment and buffer it to list
    """

    def add_fragment(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        self.fragments.append(tokenized_fragment)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    """
    Uses Word2Vec to create a vector for each fragment
    Gets a vector for the fragment by combining token embeddings
    Number of tokens used is min of number_of_tokens and 100
    """

    #word2vector将每个fragment转为词向量，最后返回一个fragment的词向量
    def vectorize(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        vectors = np.zeros(shape=(100, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_fragment), 100)):
                vectors[100- 1 - i] = self.embeddings[tokenized_fragment[len(tokenized_fragment) - 1 - i]]
        else:
            for i in range(min(len(tokenized_fragment), 100)):
                vectors[i] = self.embeddings[tokenized_fragment[i]]
        return vectors

    """
    Done adding fragments, then train Word2Vec model
    Only keep list of embeddings, delete model and list of fragments
    """
    
    #用word2vector来训练词向量
    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        model = Word2Vec(self.fragments, min_count=1, vector_size=self.vector_length, sg=0)  # sg=0: CBOW; sg=1: Skip-Gram
        self.embeddings = model.wv
        del model
        del self.fragments
