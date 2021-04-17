from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, accuracy_mse, RMSELoss
from utils import load_data
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='cora', help='Data name to run with')
parser.add_argument('--train_ratio', type=float, default=0.05, help='The ratio of dataset for training')
parser.add_argument('--public_splitting', type=bool, default=False, help='Use the public splitting as in Yang 2016 for citation dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--early_stopping', type=int, default=50, help='Patience of early stopping')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--replicates', type=int, default=10, help='Number of experiment replicates')
parser.add_argument('--saved_name', type=str, default='sample_run.txt', help='The saved file name for one run')
parser.add_argument('--task', type=str, default='classification', help='task - classification or regression')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# -------------------------------------------
# Helper to run experiment
# -------------------------------------------
def experimenter(data_name='cora', train_ratio=0.03, cuda=True, random_seed=42, hidden=16, dropout_ratio=0.5,
                 learning_rate=0.01, weight_decay=5e-4, num_epochs=100, early_stopping=30, task='classification', public_splitting=False):
    # helper function to run epxeriment 
    if data_name in ['cora', 'citeseer', 'pubmed']:
        print("Loading Classification Datasets")
        Tmat, eadj, edge_name, edge_feature_dict, adj, features, edge_features, labels, idx_train, idx_val, idx_test = tqdm(load_data(data_name=data_name,
                                                                                                                            train_ratio=train_ratio,
                                                                                                                            public_splitting=public_splitting)) 
        model = GCN(nfeat_v=features.shape[1], nfeat_e=edge_features.shape[1], nhid=hidden,
                    nclass=labels.max().item() + 1, dropout=dropout_ratio)

    else:
        ValueError("The input data is not supported! ")
    print(">" * 100)
    print("Loaded and preprocessed the graph data! ")
    print(">" * 100) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if cuda:
        torch.cuda.manual_seed(random_seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if cuda:
        model.cuda()
        Tmat = Tmat.cuda()
        eadj = eadj.cuda()
        adj = adj.cuda()
        features = features.cuda()
        edge_features= edge_features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        # pooling = pooling.cuda()
        # node_count = node_count.cuda()

    if task == "classification":
        criteria = F.nll_loss
        acc_measure = accuracy
    elif task == "regression":
        criteria = torch.nn.L1Loss
        acc_measure = RMSELoss 
    # ---------------------------------------
    # training function
    # ---------------------------------------
    # count_time = 0

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_features, eadj, adj, Tmat, task)

        loss_train = criteria(output[idx_train], labels[idx_train])
        acc_train = acc_measure(output[idx_train], labels[idx_train])
       
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, edge_features, eadj, adj, Tmat,task)

        loss_val = criteria(output[idx_val], labels[idx_val])
        acc_val = acc_measure(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_val.item()

    # -------------------------------------------
    # testing function
    # -------------------------------------------
    def test():
        model.eval()
        output = model(features, edge_features, eadj, adj, Tmat, task)
        loss_test = criteria(output[idx_test], labels[idx_test])
        acc_test = acc_measure(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    # Train model
    t_total = time.time()
    val_watch = []
    for epoch in range(num_epochs):
        val_watch.append(train(epoch))
        test()
        if epoch > early_stopping and val_watch[-1] > np.mean(val_watch[-(early_stopping + 1):-1]):
            print("Early stopping...")
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Printing the weights : ")

    return test()


if __name__ == '__main__':
    # Look for your absolute directory path
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    file_path = "../results/citeseer_1pct/" + args.saved_name
    with open(file_path, "w") as text_file:
        for i in range(args.replicates):
            np.random.seed(args.seed)
            current_torch_seed = args.seed + int(i * 10)
            torch.manual_seed(args.seed + int(i *10))
            if args.cuda:
                torch.manual_seed(args.seed + int(i * 10))
                torch.backends.cudnn.deterministic=True
                torch.cuda.manual_seed(args.seed + int(i * 10))
            print("=" * 100)
            print("Start the ", str(i + 1), "th replicate!")
            print("=" * 100)
            header = tabulate([['task', args.task],
                              ['data_name', args.data_name],
                              ['current_torch_seed', current_torch_seed],
                              ['train_ratio', args.train_ratio],
                              ['num_hidden', args.hidden],
                              ['dropout_ratio', args.dropout],
                              ['learning_rate', args.lr],
                              ['num_epochs', args.epochs],
                              ['early_stopping', args.early_stopping]
                              ], headers=['Argument', 'Value'])
            print(header) 
            tmp = experimenter(data_name=args.data_name, train_ratio=args.train_ratio, cuda=args.cuda,
                               random_seed=args.seed, hidden=args.hidden, dropout_ratio=args.dropout,
                               learning_rate=args.lr, weight_decay=args.weight_decay, num_epochs=args.epochs,
                               early_stopping=args.early_stopping, task=args.task, public_splitting=args.public_splitting)
            print(tmp)
            text_file.write(str(header) + '\n')
            text_file.write('\n' + 'Accuracy on test set: ' + '\n')
            text_file.write(str(tmp) + '\n')
            print("=" * 100)
            print("Finished the ", str(i + 1), "th replicate!")
            print("=" * 100)

