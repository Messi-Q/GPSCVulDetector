import numpy as np


def encode_one_hot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_one_hot


class vec2onehot:
    varOP_sentence = []
    nodeOP_sentence = []
    var_sentence = []
    varOP_vectors = {}
    nodeOP_vectors = {}
    var_vectors = {}

    # infinite loop
    loops = ['pattern1', 'pattern2', 'pattern3']
    # reentrancy
    reentrancy = ['pattern1', 'pattern2', 'pattern3']
    # timestamp
    timestamp = ['pattern1', 'pattern2', 'pattern3']

    def __init__(self):
        for i in range(len(self.timestamp)):
            self.var_sentence.append(i + 1)
        for i in range(len(self.loops)):
            self.varOP_sentence.append(i + 1)
        for i in range(len(self.reentrancy)):
            self.nodeOP_sentence.append(i + 1)
        self.var_dict = dict(zip(self.timestamp, self.var_sentence))
        self.varOP_dict = dict(zip(self.loops, self.varOP_sentence))
        self.nodeOP_dict = dict(zip(self.reentrancy, self.nodeOP_sentence))
        self.timestamp2vec()
        self.loops2vec()
        self.reentrancy2vec()

    def output_vec(self, vectors):
        for node, vec in vectors.items():
            print("{} {}".format(node, ' '.join([str(x) for x in vec])))

    def timestamp2vec(self):
        for word, index in self.var_dict.items():
            node_array = np.zeros(len(self.timestamp), dtype=int)
            self.var_vectors[word] = node_array
            self.var_vectors[word][index - 1] = 1.0

    def timestamp2vecEmbedding(self, var):
        return self.var_vectors[var]

    def loops2vec(self):
        for word, index in self.varOP_dict.items():
            node_array = np.zeros(len(self.loops), dtype=int)
            self.varOP_vectors[word] = node_array
            self.varOP_vectors[word][index - 1] = 1.0

    def loops2vecEmbedding(self, varOP):
        return self.varOP_vectors[varOP]

    def reentrancy2vec(self):
        for word, index in self.nodeOP_dict.items():
            node_array = np.zeros(len(self.reentrancy), dtype=int)
            self.nodeOP_vectors[word] = node_array
            self.nodeOP_vectors[word][index - 1] = 1.0

    def reentrancy2vecEmbedding(self, verOP):
        return self.nodeOP_vectors[verOP]
