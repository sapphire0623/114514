import torch
import torch.nn as nn

#定义网络
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,10),
        )
        
    def forward(self,x):
        flatten = nn.Flatten()
        x = flatten(x)
        x = self.net(x)
        return x
    
model = TestNet().cuda()
#print(model)

#优化器
from torch.optim import Adam
optimizer = Adam(model.parameters(),lr=1e-3)

#损失函数
import torch.nn.functional as F
loss_func = F.cross_entropy

#训练
def trainer(batch, model, optimizer, loss_func, device):
    model.train()

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    y_hat = model(x)

    loss = loss_func(y_hat,y)

    loss.backward()
    optimizer.step()
    
    predictions = torch.argmax(y_hat, dim=1)
    correct_predictions = (predictions == y).sum().item()
    return loss.item() / y.shape[0], correct_predictions


#验证
def validate(val_loader, model, loss_func, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            total_loss += loss.item() / y.size(0)
            
            predictions = torch.argmax(y_hat, dim=1)
            correct_predictions += (predictions == y).sum().item()
            total_samples += y.size(0)
            
    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    print(f"Validation Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    return average_loss, accuracy
    

#数据集
from torchvision.datasets import MNIST
import torchvision.transforms as T

my_transform = T.Compose([T.ToTensor(),T.Normalize((0.5,), (0.5,))])

train_dataset = MNIST(root='./data', train=True, transform=my_transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=my_transform, download=True)

#数据读取
from torch.utils.data import DataLoader
batch_size = 512
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# 新建log文件
log_file_name='231206exp_output'
f = open(log_file_name+'.txt','w',encoding='utf-8')

#开始训练
device = "cuda"
total_ep = 20


for ep in range(total_ep):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in train_loader:
        loss, correct_predictions = trainer(batch, model, optimizer, loss_func, device)
        total_loss += loss
        total_correct += correct_predictions
        total_samples += batch[1].shape[0]
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    print(f"Epoch: {ep} Training Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
    f.write(f"Epoch: {ep} Training Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%\n")
        
    if (ep+1) % 5 == 0:
        validate(test_loader, model, loss_func, device)

f.close()
