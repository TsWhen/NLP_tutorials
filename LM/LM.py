import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

########################################
# 神经网络语言模型：输入层、隐藏层、输出层，
# lookup层到隐藏层的权值矩阵H和偏置矩阵B
# 隐藏层到输出层的权值矩阵U和偏置矩阵D
# 输入层神经元数对应词的滑动窗口
########################################
class LMNet(nn.Module):

    def __init__(self,vocab_size,hidden_size,embedding_size,steps):

        super(LMNet,self).__init__()

        self.encoder = nn.Embedding(vocab_size,embedding_size)
        self.H = nn.Parameter(torch.randn(steps*embedding_size,hidden_size).type(torch.FloatTensor))
        self.b = nn.Parameter(torch.randn(hidden_size).type(torch.FloatTensor))
        self.U = nn.Parameter(torch.randn(hidden_size,vocab_size).type(torch.FloatTensor))
        self.d = nn.Parameter(torch.randn(vocab_size).type(torch.FloatTensor))
        self.W = nn.Parameter(torch.randn(steps * embedding_size,vocab_size).type(torch.FloatTensor))

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.steps = steps
        self.embedding_size = embedding_size

    def forward(self, input):

        X = self.encoder(input)
        X = X.view(-1,self.steps * self.embedding_size)
        output = torch.tanh(self.b + torch.mm(X,self.H))
        output = self.d + torch.mm(output,self.U) + torch.mm(X,self.W)
        return output


class LMModel(nn.Module):

    def __init__(self,vocab_size,hiddensize,embedding_size,steps,word_dict):

        super(LMModel,self).__init__()

        self.lmnet = LMNet(vocab_size,hiddensize,embedding_size,steps)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.lmnet.parameters(),lr=0.001)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.word_dict = word_dict

    def train(self,epoch_num,data):

        input_batch,target_batch = self.get_batch(data)
        input_batch = Variable(torch.LongTensor(input_batch)).to(self.device)
        target_batch = Variable(torch.LongTensor(target_batch)).to(self.device)

        self.lmnet = self.lmnet.to(self.device)

        for epoch in range(epoch_num):

            self.optimizer.zero_grad()
            pred = self.lmnet(input_batch)
            loss = self.criterion(pred,target_batch)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    def predict(self,data):

        data = Variable(torch.LongTensor(data)).to(self.device)
        self.lmnet.to(self.device)
        pred = self.lmnet(data)
        return pred

    def get_batch(self,data):

        input_batch = []
        target_batch = []

        for seq in data:

            word_list = seq.strip().split()
            input_seq = [self.word_dict[word] for word in word_list[:-1]]
            target = word_dict[word_list[-1]]

            input_batch.append(input_seq)
            target_batch.append(target)

        return  input_batch,target_batch

if __name__ == '__main__':

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}

    model = LMModel(len(word_list),2,2,2,word_dict)
    model.train(5000,sentences)
    input_batch,_ = model.get_batch(sentences)
    # Predict
    predict = model.predict(input_batch)
    print(predict)
    predict = model.predict(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])