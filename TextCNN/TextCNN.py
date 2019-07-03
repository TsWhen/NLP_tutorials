import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

##########################################
# 使用CNN进行文本分类
# 输入是一个二维tensor(经过embedding层)，每一行表示一个词的向量
# 卷积核宽度一般和词向量维度一样，高度为超参数
# 池化层做max-overtime-pooling，然后将得到的值做concate
# 池化层后接上全连接层和softmax层做分类，使用L2以及Dropout防止过拟合
##########################################




if __name__ == '__main__':

    pass