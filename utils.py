import pickle
import copy
import numpy as np

def batch_generator(arr,n_seqs,n_steps):
    """
    生成批次数据
    :param arr:
    :param n_seqs:
    :param n_steps:
    :return:
    """
    arr=copy.copy(arr)
    batch_size=n_seqs*n_steps
    n_batches=int(len(arr)/batch_size)
    arr=arr[:batch_size*n_batches]
    arr=arr.reshape((n_seqs,-1))
    while True:
        np.random.shuffle(arr)
        for n in range(0,arr.shape[1],n_steps):
            x=arr[:,n:n+n_steps]
            y=np.zeros_like(x)
            y[:,:-1],y[:,-1]=x[:,1:],x[:,0]
            yield x,y




class TextConverter(object):

    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as in_data:
                self.vocab = pickle.load(in_data)
        else:
            vocab = set(text)
            print(len(vocab))
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0

            for word in text:
                vocab_count[word] += 1

            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)

            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]

            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
    @property
    def vocab_size(self):
        """
        词汇表大小 +1 因为<unk>
        :return:
        """
        return len(self.vocab) + 1

    def word_to_int(self, word):
        """
        根据word转为索引index
        :param word:
        :return:
        """
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]

        else:
            return len(self.vocab)

    def int_to_word(self, index):
        """
        根据索引转为word
        :param index:
        :return:
        """
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        """
        将文本根据word找到索引，然后转为数组
        :param text:
        :return:
        """
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        """
        根据数组转为文本
        :param arr:
        :return:
        """
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        """
        序列化词汇
        :param filename:
        :return:
        """
        with open(filename, 'wb') as out_data:
            pickle.dump(self.vocab, out_data)
