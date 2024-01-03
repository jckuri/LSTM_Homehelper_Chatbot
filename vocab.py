import torch

def separate_ending_characters(word):
    words = []
    word = word.strip()
    if len(word) == 0: return []
    code = ord(word[-1])
    if ord('a') <= code <= ord('z') or ord('0') <= code <= ord('9'):
        words.append(word)
    else:
        words += separate_ending_characters(word[:-1])
        words.append(word[-1])
    return words


def sentence_to_word_list(sentence):
    words = []
    for word in sentence.split(' '):
        words += separate_ending_characters(word)
    return words
    

def sentence_to_tensor(vocab, sentence, device):
    indices = []
    indices.append(SOS)
    for word in sentence_to_word_list(sentence):
        if word in vocab.word_to_index:
            indices.append(vocab.word_to_index[word])
    indices.append(EOS)
    return torch.Tensor(indices).long().view(-1, 1).to(device)
    

SOS = 0
EOS = 1
SOS_string = '<SOS>'
EOS_string = '<EOS>'

class Vocab:
    
    def __init__(self):
        self.word_to_index = {SOS_string: SOS, EOS_string: EOS}
        self.index_to_word = {SOS: SOS_string, EOS: EOS_string}
        self.word_count = len(self.word_to_index)
        
    
    def load(self, dataset):
        n_samples = len(dataset)
        for question, answer in dataset:
            for word in sentence_to_word_list(question):
                self.add_word(word)
            for word in sentence_to_word_list(answer):
                self.add_word(word)
        print(f'n_samples={n_samples}, word_count={self.word_count}')
    
        
    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.word_count
            self.index_to_word[self.word_count] = word
            self.word_count += 1