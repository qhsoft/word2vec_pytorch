import numpy
from collections import deque
numpy.random.seed(12345)


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count=2,window_size=20):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.window_size=window_size

        self.word_pair_catch = deque()

        self.word_pairs=None
        self.word_pair_pos=0
        self.word_pair_count=0

        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        low_word_count=0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                low_word_count+=1
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
        print('low word count:',low_word_count)

    def init_sample_table(self):
        sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            sample_table += [wid] * int(c)
        self.sample_table = numpy.array(sample_table)

    def init_pairs(self,window_size):
        with open(self.input_file_name) as input_file:
            while 1:
                sentence = input_file.readline()
                if sentence is None or sentence == '':
                    break
                word_ids = []
                for word in sentence.strip().split(' '):
                    if word!='':
                        try:
                            word_ids.append(self.word2id[word])
                        except:
                            continue

                for i, u in enumerate(word_ids):
                    for j, v in enumerate(word_ids[max(i - window_size, 0):i + window_size]):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        self.word_pairs.append([u, v])
        self.word_pair_pos=0
        self.word_pairs=numpy.array(self.word_pairs)
        self.word_pair_count=len(self.word_pairs)
        print('init pairs finished, pair count:',self.word_pair_count)
    
    #@profile
    def get_batch_pairs(self, batch_size):
        """
        本打算用预加载的方法，先将所有pair准备好放在内存中进行加速，但实际效果居然更慢
        不知道为什么。

        所以此函数不会被调用，Python也会根据参数数目自动选择合适的函数进行调用的。
        """
        if self.word_pairs is None:
            self.init_pairs(self.window_size)

        batch_pairs = []
        for _ in range(batch_size):
            p=self.word_pairs[self.word_pair_pos]
            self.word_pair_pos=(self.word_pair_pos+1)%self.word_pair_count
            #print(p)
            batch_pairs.append(p)
        return batch_pairs

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            p=self.word_pair_catch.popleft()
            #print(p)
            batch_pairs.append(p)
        return batch_pairs

    #@profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./graph_list_out.txt')
    print(a.evaluate_pair_count(20))

if __name__ == '__main__':
    test()
