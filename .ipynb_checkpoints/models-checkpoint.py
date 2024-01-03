import torch
import random
from vocab import *

class Encoder(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # self.embedding provides a vector representation of the inputs to our model
        self.embedding = torch.nn.Embedding(self.input_size, self.hidden_size)
        
        # self.lstm, accepts the vectorized input and passes a hidden state
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)
        
    
    def forward(self, i, h, c):
        
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state
                c, the cell state
        '''
        i2 = self.embedding(i)
        i2 = i2.view(1, 1, -1)
        o, (h, c) = self.lstm(i2, (h, c))
        
        return o, h, c


class Decoder(torch.nn.Module):
      
    def __init__(self, hidden_size, output_size):
        
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # self.embedding provides a vector representation of the target to our model
        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        
        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)

        # self.ouput, predicts on the hidden state via a linear output layer     
        self.output = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim = 1)
        
        
    def forward(self, i, h, c):
        
        '''
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        '''
        
        i2 = self.embedding(i)
        i2 = i2.view(1, 1, -1)
        o, (h, c) = self.lstm(i2, (h, c))
        o = self.softmax(self.output(o[0]))
        return o, h, c


class SequenceToSequence(torch.nn.Module):
    
    def __init__(self, vocab, hidden_size):
        
        super(SequenceToSequence, self).__init__()
        
        self.train_on_gpu = True
        self.device = torch.device("cuda" if self.train_on_gpu else "cpu")
        
        self.vocab = vocab
        self.hidden_size = hidden_size
        
        self.encoder = Encoder(self.vocab.word_count, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.vocab.word_count)
    
    
    def forward(self, src, trg, src_length, trg_length, teacher_forcing_ratio = 0.5):
        
        output_list = []
        
        # The number of LSTM layers is 1
        h = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        c = torch.zeros([1, 1, self.hidden_size]).to(self.device)  
        
        for i in range(src_length):
            encoder_output, h, c = self.encoder(src[i], h, c)

        decoder_input = torch.Tensor([[SOS]]).long().to(self.device)
        
        for i in range(trg_length):
            decoder_output, h, c = self.decoder(decoder_input, h, c)
            output_list.append(decoder_output)
            
            if self.training:
                decoder_input = trg[i] if random.random() < teacher_forcing_ratio else decoder_output.argmax(1)
            else:
                _, top_index = decoder_output.data.topk(1)
                decoder_input = top_index.squeeze().detach()
        
        return {'decoder_output': output_list}