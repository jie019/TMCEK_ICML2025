import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim


N_classes=5
num_layers=2
N_around=4
model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% network definition
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
    
#%% data manager
def making_batch(F_in,L_in,n_batch,prob):
    data_1=[]
    data_2=[]
    label=[]
    temp=np.arange(N_around+1,F_in.shape[0]-N_around-1)
    selected=np.random.choice(temp, size=n_batch, p=prob)
    for i in selected:
        label.append(L_in[i])
        data_1.append(F_in[i-N_around:i+1,:])
        data_2.append(F_in[i:i+N_around+1,:])
    return data_1,data_2,label

def cm__(target,lebels,n=5):
    cm=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            cm[i][j]=sum((target==i)&(lebels==j))
    return cm


#%% initial
def set_labels(labels):
    labels=np.array(labels)
    labels_=set(labels)
    probabilities=np.ones_like(labels,dtype=np.float64)
    for c in range(len(labels_)):
        count=np.sum(labels==c)
        probabilities[labels==c]=1/count
    probabilities=probabilities/sum(probabilities)
    return labels, probabilities

j=0
train_file_name = f'Data_for_LSTM{j}.data'
temp=torch.load(train_file_name)
OUT=temp['OUT']
train_labels=temp['train_labels']
train_labels,train_prob=set_labels(train_labels)
prob=train_prob[N_around+1:OUT.shape[0]-N_around-1]
prob=prob/sum(prob)
minibach_size=16
running_loss = 0.0
count=0
n_iter=50000
saving_iter=1000
lr_sch_iter=5000
print_loss_iter=1000

train_loss_=[]

net=Net_LSTM()
net=net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01/minibach_size)

for i in range(n_iter):
    data_1,data_2,label=making_batch(F_in=OUT,L_in=train_labels,n_batch=minibach_size,prob=prob)
    label= torch.tensor(label).type('torch.LongTensor')
    data_1=torch.tensor(data_1).type('torch.FloatTensor')
    data_2=torch.tensor(data_2).type('torch.FloatTensor')
    data_1.transpose_(0,1)
    data_2.transpose_(0,1)
    data_1, data_2, label = data_1.to(device), data_2.to(device), label.to(device)
    
    optimizer.zero_grad()
    outputs = net(data_1,data_2)
    loss = criterion(outputs,label)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
    if i % print_loss_iter == (print_loss_iter-1):   
        print(str(running_loss / print_loss_iter))
        train_loss_.append(running_loss / print_loss_iter)
        running_loss = 0.0
        
    if i % lr_sch_iter==(lr_sch_iter-1):
        optimizer.param_groups[0]['lr']=.9*optimizer.param_groups[0]['lr']
        print(optimizer.param_groups[0]['lr'])
        
    if i % saving_iter == (saving_iter-1):
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss':train_loss_}
                    ,model_dir+'LSTM_'+str(i))




