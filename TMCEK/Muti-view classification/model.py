import torch.nn as nn
import torch
import torch.nn.functional as F

def compute_gini(data, num_classes,dim=1):
    probs = data / (torch.sum(data, dim=dim, keepdim=True) + 1e-8)
    gini = 1 - torch.sum(probs**2, dim=dim)
    gini = gini.unsqueeze(1).repeat(1, num_classes)
    return gini



class TMCEK(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(TMCEK, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            gini_a= compute_gini(evidence_a, self.num_classes)
            epsilon_a = (gini_a + 1) / 2
            gini= compute_gini(evidences[i], self.num_classes)
            epsilon = (gini + 1) / 2
            evidence_a = (epsilon_a*evidences[i] + epsilon*evidence_a) / (epsilon + epsilon_a)
        return evidences, evidence_a


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
