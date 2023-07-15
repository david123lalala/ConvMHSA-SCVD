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


class FragmentVectorizer:
    def __init__(self, vector_length):
        self.fragments = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0

    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0

        while i < len(line):
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
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
            else:
                w.append(line[i])
                i += 1

        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))
    
    @staticmethod
    def tokenize_fragment(fragment):
        tokenized = []
        function_regex = re.compile('function(\d)+')
        backwards_slice = False
        for line in fragment:
            tokens = FragmentVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    def add_fragment(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        self.fragments.append(tokenized_fragment)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

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
    def train_model(self):
        model = Word2Vec(self.fragments, min_count=1, vector_size=self.vector_length, sg=0)  # sg=0: CBOW; sg=1: Skip-Gram
        self.embeddings = model.wv
        del model
        del self.fragments
