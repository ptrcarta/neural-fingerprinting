from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import dill as pickle
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(epoch, args, model, optimizer, data_loader, ds_name = None):

    fingerprint_accuracy = []

    loss_n = torch.nn.MSELoss()

    for batch_idx, (x, y) in enumerate(data_loader):
        model.train()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        optimizer.zero_grad()
        real_bs = y.size(0)
        loss_func = nn.CrossEntropyLoss()

        ## Add loss for (y+dy,model(x+dx)) for each sample, each dx
        x_net = x

        logits_net = model(x_net)
        output_net = F.log_softmax(logits_net)

        yhat = output_net[0:real_bs]
        logits = logits_net[0:real_bs]
        logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)
        loss_fingerprint_y = 0
        loss_vanilla = loss_func(yhat, y)

        loss = loss_vanilla
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss vanilla: {:.3f} Total Loss: {:.3f}'.format(
                epoch, batch_idx * len(x), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss_vanilla.item(),
                loss.item()))

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--ds-name', type=str, default = 'mnist',
                    help='Dataset -- mnist, cifar, miniimagenet')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--num-dx', type=int, default=5)
parser.add_argument('--num-class', type=int, default=10)

parser.add_argument('--name', default="dataset-name")
parser.add_argument("--data-dir", default='MNIST_data/')
parser.add_argument("--log-dir")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=True, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

from mnist.model import CW_Net as Net
model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Args:", args)

for epoch in range(1, args.epochs + 1):
    train(epoch, args, model, optimizer, train_loader, args.ds_name)
