import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset


N_classes=5
data_directory='data/Sleep-EDF 20/'
model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% define wave layer
class Wave(nn.Module):
    def __init__(self,n_filt,n_time,n_in=1,strid=1):
        super(Wave, self).__init__()
        self.n_filt=n_filt
        self.n_time=n_time
        self.n_in=n_in
        self.strid=strid
        self.time=((torch.unsqueeze(torch.tensor(range(self.n_time)),1).t().type('torch.FloatTensor')+1-n_time/2)/100).to(device)
        self.u=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.w=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.s=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.filt=[]

    def forward(self, x):
        u=self.u.expand(self.n_filt,self.n_time)
        w=self.w.expand(self.n_filt,self.n_time)*3
        s=self.s.expand(self.n_filt,self.n_time)*5
        time=self.time.expand_as(s)
        filt=torch.exp(-3.1314*torch.abs(s)*((time-u)**2))*torch.cos(2*3.1415*w*10*time)
        self.filt=filt.to(device)
        filt=torch.unsqueeze(filt,1) 
        filt=filt.repeat(1,self.n_in,1)
        return F.conv1d(x,filt,stride=self.strid)
    def return_filt(self):
        return self.filt
        

def compute_gini(data, num_classes, dim=1):
    probs = data / (torch.sum(data, dim=dim, keepdim=True) + 1e-8)  # 归一化为概率分布
    gini = 1 - torch.sum(probs ** 2, dim=dim)  # 计算基尼系数
    gini = gini.unsqueeze(1).repeat(1, num_classes)  # 扩展基尼系数以匹配类别维度
    return gini


class Net(nn.Module):
    def __init__(self,N_eeg_wave=32,wave_time=2,fs=100):
        super(Net, self).__init__()
        self.wav1=Wave(N_eeg_wave,wave_time*fs)
        self.pool = nn.MaxPool1d(3)
        self.mixing = nn.Linear(N_eeg_wave,256)
        self.conv1 = nn.Conv1d(256, 64, 3)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 3,stride=2)
        self.conv3_bn = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3,stride=2)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, 3,stride=2)
        self.conv5_bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256*12, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, N_classes)
        self.en1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),  # 从4个通道开始的2D卷积
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.Dropout(0.1))
        self.en2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)))
        self.en3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)))
        self.en4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 2), padding=(1, 2))
        )
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, N_classes)
        self.avg = nn.AdaptiveAvgPool2d((1))
        self.dr2 = nn.Dropout(p=0.6)
        self.dr1 = nn.Dropout(p=0.6)


    def forward(self, x1,tf):
        evidences = dict()
        x1 = F.normalize(x1)
        x1 =F.relu(self.wav1(x1))
        x1=torch.transpose(x1, 1, 2)
        x1=self.mixing(x1)
        x1=torch.transpose(x1, 2, 1)
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.conv1_bn(x1)
        x1 =self.pool(F.relu(self.conv2(x1)))
        x1 = self.conv2_bn(x1)
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = self.conv3_bn(x1)
        x1 = (F.relu(self.conv4(x1)))
        x1 = self.conv4_bn(x1)
        x1 = (F.relu(self.conv5(x1)))
        x1 = self.conv5_bn(x1)
        x = x1.view(-1, self.num_flat_features(x1))
        x=self.dr2(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.softplus(self.fc3(x))

        tf1 = self.en1(tf)
        tf1 = self.en2(tf1)
        tf1 = self.en3(tf1)
        tf1 = self.en4(tf1)
        tf1 = self.avg(tf1).squeeze()
        y1 = F.gelu(self.fc4(tf1))
        y1 = F.gelu(self.fc5(y1))
        y1 = F.softplus(self.fc6(y1))

        evidences[0] = y
        evidences[1] = y1
        evidence_a = evidences[0]
        for i in range(1, 2):
            gini_a = compute_gini(evidence_a, N_classes)
            epsilon_a = (gini_a + 1) / 2
            gini = compute_gini(evidences[i], N_classes)
            epsilon = (gini + 1) / 2
            evidence_a = (epsilon_a * evidences[i] + epsilon * evidence_a) / (epsilon + epsilon_a)
        return evidences,evidence_a

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

#%% initial


class CustomDataset(Dataset):
    def __init__(self, data, labels,  transform=None):
        self.data = data
        self.labels = labels
        self.data_directory = data_directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 加载数据
        data = self.data[idx]
        label = self.labels[idx]
        # 计算STFT
        tf = torch.stft(
            torch.tensor(data),
            n_fft=256,
            hop_length=128,
            win_length=256,
            window=torch.hann_window(256),
            return_complex=False
        ).permute(0, 3, 2, 1)
        tf = tf.reshape(-1, 24, 129)
        return data, label, tf

def set_labels(labels):
    labels = np.array(labels)
    labels_ = set(labels)
    probabilities = np.ones_like(labels, dtype=np.float64)
    for c in range(len(labels_)):
        count = np.sum(labels == c)
        probabilities[labels == c] = 1 / count
    probabilities = probabilities / sum(probabilities)
    return labels, probabilities


filename = os.listdir(data_directory)
filenames=[name for name in filename if '.npz' in name]
filenames.sort()
acc_score=[]
F1_score=[]
kappa_score=[]
n1_f1_score=[]
j=0
test_filenames = [f for f in filenames if str(j).zfill(2)==f[3:5]]
train_filenames = [f for f in filenames if str(j).zfill(2)!=f[3:5]]
train_labels = []
train_data = []

if j != 13:
    test_labels = []
    test_data = []
    for test_file in test_filenames:
        data = np.load(os.path.join(data_directory, test_file))
        data1 = data['x']
        label = data['y']
        test_labels.append(label)
        test_data.append(data1)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
else:
    data = np.load(os.path.join(data_directory, test_filenames[0]))
    test_data = data['x']
    test_labels = data['y']
for train_file in train_filenames:
    data = np.load(os.path.join(data_directory, train_file))
    data1=data['x']
    label=data['y']
    train_labels.append(label)
    train_data.append(data1)
train_data = np.concatenate(train_data, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
train_labels, train_prob = set_labels(train_labels)
test_labels, test_prob = set_labels(test_labels)
train_dataset = CustomDataset(train_data, train_labels, data_directory)
test_dataset = CustomDataset(test_data, test_labels, data_directory)


temp=torch.load(model_dir+'model_366999', map_location=lambda storage, loc: storage)
net=Net()
net=net.to(device)
net.load_state_dict(temp['model_state_dict'])
net.eval()


def making_batch(data, labels, prob, n_batch=12, ite=1):
        t_data = []
        label = []
        tf_data = []
        selected = np.arange(n_batch * ite, n_batch * (ite + 1))
        for i in selected:
            data1=data[i]
            tf = torch.stft(
                torch.tensor(data1),
                n_fft=256,
                hop_length=128,
                win_length=256,
                window=torch.hann_window(256),
                return_complex=False
            ).permute(0, 3, 2, 1)
            tf = tf.reshape(-1, 24, 129)
            label.append(labels[i])
            t_data.append(data1)
            tf_data.append(tf)
        return t_data, label, tf_data


test_minibach_size = 16
n_iter = len(train_labels) / test_minibach_size
OUT = 0
cm = np.zeros([5, 5])
for i in range(int(n_iter)):
    if i % 100 == 99:
        print(str(i + 1) + '/' + str(int(n_iter)))
    data, label, tf = making_batch(data=train_data, labels=train_labels, prob=None, n_batch=test_minibach_size,ite=i)
    label = torch.tensor(label).type('torch.LongTensor')
    data = torch.tensor(data).type('torch.FloatTensor')
    data, label = data.to(device), label.to(device)
    tf = np.array(tf)
    tf = torch.from_numpy(tf).type('torch.FloatTensor').to(device)
    outputs, evidences = net(data[:, 0, :].unsqueeze(1), tf)
    if i == 0:
        OUT = outputs.cpu().detach().numpy()
    else:
        OUT = np.concatenate((OUT, outputs.cpu().detach().numpy()), axis=0)
train_file_name = f'Data_for_LSTM{j}.data'
torch.save({'OUT': OUT, 'train_labels': train_labels}, train_file_name)
n_iter = len(test_labels) / test_minibach_size
OUT = 0
cm = np.zeros([5, 5])
for i in range(int(n_iter)):
    if i % 20 == 19:
        print(str(i + 1) + '/' + str(int(n_iter)))
    data, label, tf = making_batch(data=test_data, labels=test_labels, prob=None, n_batch=test_minibach_size,ite=i)
    label = torch.tensor(label).type('torch.LongTensor')
    data = torch.tensor(data).type('torch.FloatTensor')
    data, label = data.to(device), label.to(device)
    tf = np.array(tf)
    tf = torch.from_numpy(tf).type('torch.FloatTensor').to(device)
    outputs, evidences = net(data[:, 0, :].unsqueeze(1),tf)
    cm += cm__(label.cpu().numpy(), torch.argmax(outputs, 1).cpu().numpy())
    if i == 0:
        OUT = outputs.cpu().detach().numpy()
    else:
        OUT = np.concatenate((OUT, outputs.cpu().detach().numpy()), axis=0)
print((cm).astype(int))
acc = np.sum(np.diag(cm)) / np.sum(cm)
print(acc)
test_file_name = f'Data_for_LSTM_test{j}.data'
torch.save({'OUT': OUT, 'test_labels': test_labels,  'CM': cm}, test_file_name)


