import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import Scene, HandWritten, PIE,Cal101_20,CNIST,CUB
from loss_function import get_loss
from model import TMCEK
import os
np.set_printoptions(precision=4, suppress=True)
torch.autograd.set_detect_anomaly(True)
def calculate_mean_and_std(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    import math
    std = math.sqrt(variance)
    return mean, std


def normal(args):
    dataset = PIE()
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    test_time = 5
    results = []
    for test_t in range(test_time):
        index = np.arange(num_samples)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

        model = TMCEK(num_views, dims, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gamma = 1
        model.to(device)
        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            print(f'====> {epoch}')
            for X, Y, indexes in train_loader:
                for v in range(num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)
                evidences, evidence_a = model(X)
                loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, gamma, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            average_loss = epoch_loss / num_batches
            print(f'Average Loss for Epoch {epoch}: {average_loss}')
        model.eval()
        num_correct, num_sample = 0, 0
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            with torch.no_grad():
                evidences, evidence_a = model(X)
                _, Y_pre = torch.max(evidence_a, dim=1)
                num_correct += (Y_pre == Y).sum().item()
                num_sample += Y.shape[0]
        acc = num_correct / num_sample
        print(acc)
        results.append(acc)
    mean, stdDev = calculate_mean_and_std(results)
    print(results)
    print('mean = {:.4f}'.format(mean))
    print('std = {:.4f}'.format(stdDev))

    pass


def conflict(args):
    dataset = PIE()
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    # create a test set with conflict instances
    dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)
    model = TMCEK(num_views, dims, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = 1
    model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        print(f'====> {epoch}')
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X)
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, gamma, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        average_loss = epoch_loss / num_batches
        print(f'Average Loss for Epoch {epoch}: {average_loss}')
    model.eval()
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    print('====> acc: {:.4f}'.format(num_correct / num_sample))
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')


    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    # conflict(args)
    normal(args)
