import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
from torch.utils.data import DataLoader


N_classes=5
num_layers=2
N_around=4
model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% define wave layer

class Net_LSTM(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        self.lstm_forward=nn.LSTM(N_classes,N_classes*2,num_layers=num_layers)
        self.lstm_backward=nn.LSTM(N_classes,N_classes*2,num_layers=num_layers)
        self.fc=nn.Linear(4*N_classes, N_classes)
        self.dr=nn.Dropout(p=0.5)
    def forward(self, x_for,x_back):
        _, (x_for,_)=self.lstm_forward(x_for)
        _, (x_back,_)=self.lstm_backward(x_back)
        x_for=x_for[-1,:]
        x_back=x_back[-1,:]
        x=torch.cat([x_for,x_back],1)
        x=self.dr(x)   
        x=self.fc(x)
        return x

def one_hot(label,batch_size,n_out):
    oo=np.zeros([batch_size,n_out])
    for i in range(batch_size):
        oo[i,label[i]]=1
    return oo

def cm__(target,lebels,n=5):
    cm=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            cm[i][j]=sum((target==i)&(lebels==j))
    return cm
train_loss_=[]
val_loss_=[]


def making_set(F_in, L_in):
    train_set = []
    temp = np.arange(N_around + 1, len(F_in) - N_around - 1)
    for i in temp:
        label = L_in[i]
        data_1 = F_in[i - N_around:i + 1, :]
        data_2 = F_in[i:i + N_around + 1, :]
        train_set.append((data_1, data_2, label))
    return train_set

j=0
test_file_name = f'Data_for_LSTM_test{j}.data'
temp = torch.load(test_file_name)
test_OUT = temp['OUT']
test_labels = temp['test_labels']
test_set = making_set(test_OUT, test_labels)
test_loader = DataLoader(
    test_set,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

temp=torch.load(model_dir+'model_366999', map_location=lambda storage, loc: storage)
net=Net_LSTM()
net=net.to(device)
net.load_state_dict(temp['model_state_dict'])
net.eval()

cm = np.zeros([5, 5])
out = []
L = []
for (data_1, data_2, label) in tqdm(test_loader):
    label = torch.tensor(label).type('torch.LongTensor')
    data_1 = torch.tensor(data_1).type('torch.FloatTensor')
    data_2 = torch.tensor(data_2).type('torch.FloatTensor')
    data_1.transpose_(0, 1)
    data_2.transpose_(0, 1)
    data_1, data_2, label = data_1.to(device), data_2.to(device), label.to(device)
    outputs = net(data_1, data_2)
    cm += cm__(label.cpu().numpy(), torch.argmax(outputs, 1).cpu().numpy())
    out = out + list(torch.argmax(outputs, 1).cpu().numpy())
    L = L + list(label.cpu().numpy())
out = np.array(out)
L = np.array(L)
f1 = f1_score(L, out, average="macro")
acc = np.sum(np.diag(cm)) / np.sum(cm)
kappa = cohen_kappa_score(out, L)
print(f'cm', cm.astype(int))
print(f'acc: {acc}')
print(f'f1: {f1}')
print(f'kappa: {kappa}')
wake_f1 = f1_score(L == 0, out == 0)
n1_f1 = f1_score(L == 1, out == 1)
n2_f1 = f1_score(L == 2, out == 2)
n3_f1 = f1_score(L == 3, out == 3)
rem_f1 = f1_score(L == 4, out == 4)
print(
    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
        wake_f1,
        n1_f1,
        n2_f1,
        n3_f1,
        rem_f1,
    ))