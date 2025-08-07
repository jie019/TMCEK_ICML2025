import os
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.io as sio
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


N_classes = 5
data_directory='data/Sleep-EDF 20/'
model_dir = 'Models_total/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Wave(nn.Module):
    def __init__(self, n_filt, n_time, n_in=1, strid=1):
        super(Wave, self).__init__()
        self.n_filt = n_filt
        self.n_time = n_time
        self.n_in = n_in
        self.strid = strid
        self.time = ((torch.unsqueeze(torch.tensor(range(self.n_time)), 1).t().type(
            'torch.FloatTensor') + 1 - n_time / 2) / 100).to(device)
        self.u = nn.Parameter(torch.randn(self.n_filt, 1).type('torch.FloatTensor'))
        self.w = nn.Parameter(torch.randn(self.n_filt, 1).type('torch.FloatTensor'))
        self.s = nn.Parameter(torch.randn(self.n_filt, 1).type('torch.FloatTensor'))
        self.fi = nn.Parameter(torch.randn(self.n_filt, 1).type('torch.FloatTensor'))
        self.filt = []

    def forward(self, x):
        u = self.u.expand(self.n_filt, self.n_time)
        fi = self.fi.expand(self.n_filt, self.n_time)
        w = self.w.expand(self.n_filt, self.n_time) * 3
        s = self.s.expand(self.n_filt, self.n_time) * 5
        time = self.time.expand_as(s)
        filt = torch.exp(-3.1314 * torch.abs(s) * ((time - u) ** 2)) * torch.cos(2 * 3.1415 * w * 10 * time + fi)
        self.filt = filt.to(device)
        filt = torch.unsqueeze(filt, 1)
        filt = filt.repeat(1, self.n_in, 1)
        return F.conv1d(x, filt, stride=self.strid)

    def return_filt(self):
        return self.filt

def compute_gini(data, num_classes, dim=1):
    probs = data / (torch.sum(data, dim=dim, keepdim=True) + 1e-8)  # 归一化为概率分布
    gini = 1 - torch.sum(probs ** 2, dim=dim)  # 计算基尼系数
    gini = gini.unsqueeze(1).repeat(1, num_classes)  # 扩展基尼系数以匹配类别维度
    return gini

# %% network definition
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

def one_hot(label, batch_size, n_out):
    oo = np.zeros([batch_size, n_out])
    for i in range(batch_size):
        oo[i, label[i]] = 1
    return oo


def set_labels(labels):
    labels = np.array(labels)
    labels_ = set(labels)
    probabilities = np.ones_like(labels, dtype=np.float64)
    for c in range(len(labels_)):
        count = np.sum(labels == c)
        probabilities[labels == c] = 1 / count
    probabilities = probabilities / sum(probabilities)
    return labels, probabilities


def making_batch(names, labels, prob, n_batch=12, ite=1):
    data = []
    label = []
    selected = np.arange(n_batch * (ite - 1), n_batch * ite)
    for i in selected:
        f = sio.loadmat(data_directory + names[i].replace(' ', ''))
        label.append(labels[i])
        data.append(f['SIG'])
    return data, label

def cm__(target, lebels, n=5):
    cm = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            cm[i][j] = sum((target == i) & (lebels == j))
    return cm

class CustomDataset(Dataset):
    def __init__(self, data, labels,  transform=None):
        self.data = data
        self.labels = labels
        self.data_directory = data_directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
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

j=5
filename = os.listdir(data_directory)
filenames=[name for name in filename if '.npz' in name]
filenames.sort()
test_filenames = [f for f in filenames if str(j).zfill(2)==f[3:5]]
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
test_labels, test_prob = set_labels(test_labels)
print('test subjects', j, '\n',  'test set length:', len(test_labels))
test_dataset = CustomDataset(test_data, test_labels, data_directory)
data_loader = {
    'test': DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )}

temp = torch.load(model_dir + 'Model_5_11500', map_location=lambda storage, loc: storage)
net = Net()
net = net.to(device)
net.load_state_dict(temp['model_state_dict'])
net.eval()
test_minibach_size = 20
n_iter = len(test_labels) / test_minibach_size
truths = []
preds = []
OUT = []
LABEL = []
effi_list = []
num_classes = 5
effi_list_per_class = [[] for _ in range(num_classes)]
noise_std = 10


for i,(x_t, label, tf) in enumerate(tqdm(data_loader['test'])):
    label = torch.tensor(label).type('torch.LongTensor')
    x_t = np.array(x_t)
    x_t = torch.tensor(x_t).type('torch.FloatTensor')
    x_t, label = x_t.to(device), label.to(device)
    tf = np.array(tf)
    tf = torch.from_numpy(tf).type('torch.FloatTensor').to(device)
    torch.set_grad_enabled(True)
    net.eval()
    _,outputs = net(x_t[:, 0, :].unsqueeze(1), tf)

    pred = outputs
    pred_y = torch.max(pred, dim=1)[1]
    if label.ndim == 0:
        truths.append(label.item())
    else:
        truths.extend(label.cpu().numpy().tolist())
    if pred_y.ndim == 0:
        preds.append(pred_y.item())
    else:
        preds.extend(pred_y.cpu().numpy().tolist())

truths = np.array(truths)
preds = np.array(preds)
cm = cm__(truths, preds)
acc = np.sum(np.diag(cm)) / np.sum(cm)
f1 = f1_score(truths, preds, average="macro")
kappa = cohen_kappa_score(truths, preds)
print('cm:',cm.astype(int))
print(f'acc: {acc}')
print(f'f1: {f1}')
print(f'kappa: {kappa}')
wake_f1 = f1_score(truths == 0, preds == 0)
n1_f1 = f1_score(truths == 1, preds == 1)
n2_f1 = f1_score(truths == 2, preds == 2)
n3_f1 = f1_score(truths == 3, preds == 3)
rem_f1 = f1_score(truths == 4, preds == 4)
print(
    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
        wake_f1,
        n1_f1,
        n2_f1,
        n3_f1,
        rem_f1,
    ))


