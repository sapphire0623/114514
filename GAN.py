# -*- coding: utf-8 -*-
"""
生成式对抗网络
mnist数据集
pytorch深度学习框架
生成手写数字
"""

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import os

os.chdir(os.path.dirname(__file__))

'''模型及损失函数定义'''
'模型结构'
class Generator(nn.Module):
    #生成器，将latent_size维度的数据转换为output_size维度
    def __init__(self, latent_size,  hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(latent_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x:batch_size, latent_size
        x = F.relu(self.linear(x)) #->batch_size,hidden_size
        x = torch.sigmoid(self.out(x)) #->batch_size, output_size
        return x

class Discriminator(nn.Module):
    #判别器，对input_size维度的数据输出1维，判断生成数据的真伪
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size,1)

    def forward(self, x): # x:batch_size, input_size
        x = F.relu(self.linear(x)) #->batch_size,hidden_size
        x = torch.sigmoid(self.out(x)) #->batch_size,1
        return x   
'损失函数'
#交叉熵，以衡量判别结果的与真实结果的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')


'''超参数及构造模型'''
'模型参数'
latent_size =16 #压缩后的特征维度
hidden_size = 128 #encoder和decoder中间层的维度
input_size= output_size = 28*28 #原始图片和生成图片的维度

'训练参数'
epochs = 20 #训练时期
batch_size = 32 #每步训练样本数
learning_rate = 1e-5 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

'构建模型' #如之前训练过，会导入本地已训练的模型文件
modelname = ['gan-G.pth','gan-D.pth'] #模型文件名
model_g = Generator(latent_size, hidden_size, output_size).to(device)
model_d = Discriminator(input_size, hidden_size).to(device)

optim_g = torch.optim.Adam(model_g.parameters(), lr=learning_rate)
optim_d = torch.optim.Adam(model_d.parameters(), lr=learning_rate)

try:#尝试导入本地已有模型
    model_g.load_state_dict(torch.load(modelname[0]))
    model_d.load_state_dict(torch.load(modelname[1]))
    print('[INFO] Load Model complete')
except:
    pass



''''模型训练、测试、展示''' #如上一步已导入本地模型，可省略本步骤，直接进行 模型推理
'准备mnist数据集' #(数据会下载到py文件所在的data文件夹下)
my_transform = T.Compose([T.ToTensor(),T.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='./data', train=True, transform=my_transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=my_transform, download=True)
batch_size = 512
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)



for epoch in range(epochs):   #每轮
    '模型训练'
    #每个epoch重置损失   
    Gen_loss = 0 
    Dis_loss = 0
    #获取数据
    for imgs, lbls in tqdm(train_loader,desc = f'[train]epoch:{epoch}'): #img: (batch_size,1,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0]    
        T_imgs = imgs.view(bs,input_size).to(device) #bs,input_size(28*28)
        #创造全为1（真实）与全为0（伪造）的标签待用
        T_lbl = torch.ones(bs,1).to(device) #bs,1
        F_lbl = torch.zeros(bs,1).to(device) #bs,1

        ###训练生成器
        #生成高斯噪声作为输入
        sample = torch.randn(bs,latent_size).to(device) #bs, latent_size
        #生成伪造数据
        F_imgs = model_g(sample) #bs,output_size(28*28)
        #使用判别器识别伪造数据输出判别结果
        F_Dis = model_d(F_imgs) #bs,1

        #计算损失，生成器的目标是让判别器对伪造数据的判别结果尽可能为1（真实）
        loss_g = loss_BCE(F_Dis, T_lbl) 
        #反向传播、参数优化，重置梯度
        loss_g.backward()    
        optim_g.step()
        optim_g.zero_grad()

        ###训练判别器
        #使用判别器分别判断真实图像和伪造图像
        T_Dis = model_d(T_imgs) #bs,1
        F_Dis = model_d(F_imgs.detach()) #bs,1

        #计算损失，判别器的目标是对生成器的伪造数据判别结果尽可能为0（伪造），真实数据判断结果尽可能为1（真实）
        loss_d_T = loss_BCE(T_Dis, T_lbl)
        loss_d_F = loss_BCE(F_Dis, F_lbl)
        loss_d = loss_d_T + loss_d_F        
        #反向传播、参数优化，重置梯度
        loss_d.backward()
        optim_d.step()
        optim_d.zero_grad()

        #记录总损失
        Gen_loss += loss_g.item()
        Dis_loss += loss_d.item()
    #打印该轮平均损失
    print(f'epoch:{epoch}|Train G Loss:',
          Gen_loss/len(train_loader.dataset),
          ' Train D Loss:',
          Dis_loss/len(train_loader.dataset))

    '模型测试' #如不需可省略
    model_g.eval();model_d.eval()
    #每个epoch重置损失，设置进度条    
    Gen_score = 0
    Dis_score = 0
    #获取数据
    for imgs, lbls in tqdm(test_loader,desc = f'[eval]epoch:{epoch}'):#img: (batch_size,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0] 
        T_imgs = imgs.view(bs,input_size).to(device) #batch_size,input_size(28*28)
        #模型运算
        #生成高斯噪声作为输入
        sample = torch.randn(bs,latent_size).to(device) #bs, latent_size
        #生成伪造数据
        F_imgs = model_g(sample) #bs,output_size(28*28)
        #使用判别器识别伪造数据输出判别结果
        F_Dis = model_d(F_imgs) #bs,1
        T_Dis = model_d(T_imgs) #bs,1
        #计算生成器和判别器的成功数
        #生成器成功：判别器将生成器生成图片判断为真实
        Gen_score += int(sum(F_Dis>=0.5))
        #判别器成功：将真实图片判断为真实，将生成器生成图片判断为伪造
        Dis_score += int(sum(T_Dis>=0.5)) + int(sum(F_Dis<0.5)) 

    #打印该轮成功率
    print(f'epoch:{epoch}|Test G Score:',
          Gen_score/len(test_loader.dataset),
          ' Test D Score:',
          Dis_score/len(test_loader.dataset)/2)

    model_g.train();model_d.train()


    '展示效果' #如不需可省略
    model_g.eval()
    #按标准正态分布取样来自造数据
    noise = torch.randn(1 ,latent_size).to(device)#1,latent_size
    gen_imgs = model_g(noise) #1,output_size(28*28)
    gen_imgs = gen_imgs[0].view(28,28) #28,28
    plt.matshow(gen_imgs.cpu().detach().numpy())
    plt.show()
    model_g.train()       

    '存储模型'
    torch.save(model_g.state_dict(), modelname[0])
    torch.save(model_d.state_dict(), modelname[1])


'''模型推理''' #使用经过 模型训练 的模型或者读取的本地模型进行推理
###使用生成器
#按标准正态分布取样来自造数据
sample = torch.randn(1,latent_size).to(device)#1,latent_size
gen_imgs = model_g(sample) #1,output_size(28*28)
gen_imgs = gen_imgs[0].view(28,28) #28,28
plt.matshow(gen_imgs.cpu().detach().numpy())
plt.show()

###使用判别器
#对数据集
dataset = datasets.MNIST('/data', train=False, transform=transforms.ToTensor())
#取一组数据，index为数据序数
index=0
raw = dataset[index][0].view(28,28) #raw: bs,28,28->bs,28*28
plt.matshow(raw.cpu().detach().numpy())
plt.show()
raw = raw.view(1,28*28)
result = model_d(raw.to(device))
print('该图为真概率为：',result.cpu().detach().numpy())
