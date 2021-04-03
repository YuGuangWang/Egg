#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:45:43 2020

@author: Bxin

"""
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from gnn import GNN
from pytorchtools import EarlyStopping

import numpy as np
import argparse
import os.path as osp
from QM7 import QM7, qm7_test, qm7_test_train, MyDataset
import torch.nn.functional as F
import torch_geometric.transforms as T

#%%        
def test(model, loader, device, loss_criteria=torch.nn.CrossEntropyLoss(reduction='sum')):
    model.eval()
    correct = 0.
    loss = 0.  
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()
        loss += loss_criteria(out, batch.y).item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

    
def main():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    ### below arguments are for experiment settings
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience for early stopping (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument("--wd", type=np.float, default=5e-3,
                        help="weight decay (default: 5e-3")
    ### below arguments are for network architecture
    parser.add_argument('--num_block', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--conv_type', type=str, default="gin", choices=["gin", "gcn"],
                        help='Pooling type over nodes in a graph (default: grass')   
    parser.add_argument('--norm_type', type=str, default="bn", choices=["bn", "gn"],
                        help='Pooling type over nodes in a graph (default: grass')   
    parser.add_argument('--conv_hid', type=int, default=32,
                        help='number of hidden units (default: 32)')   
    parser.add_argument('--num_conv_layer', type=int, default=2,
                        help='number of hidden mlp layers in a conv layer (default: 2)')   
    parser.add_argument('--pool_type', type=str, default="gr", choices=["max", "avg", "sum", "attention", "gr"],
                        help='Pooling type over nodes in a graph (default: grass')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio in fc layer (default: 0.5)')
    parser.add_argument('--pRatio', type=float, default=0.5,
     					help='the ratio of info preserved in the Grassmann subspace (default: 0.5)')
    parser.add_argument("--fc_dim", type=int, nargs="*", default=[64,16],
                        help="dimension of fc hidden layers (default: [64,16])") 
    ### below argumnets are for results storing
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default="results",
                        help='filename to output result (default: results)')
    
    args = parser.parse_args()    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # load dataset
    path = osp.join(osp.abspath(''), 'data', args.dataset)
    if args.dataset == 'qm7':
        dataset = QM7(path)
        num_features = 5
        num_classes = 1
        loss_criteria = F.mse_loss
        dataset, mean, std = MyDataset(dataset, num_features)
    else:
        if args.dataset == 'COLLAB':
            dataset = TUDataset(path, name=args.dataset, 
                                transform=T.OneHotDegree(max_degree = 1000))
        else:
            dataset = TUDataset(path, name=args.dataset)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        loss_criteria = F.cross_entropy
        
        
    num_train, num_val = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)
    train_graphs, val_graphs, test_graphs = random_split(dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(0))
    
    ### create results matrix
    epoch_train_loss = np.zeros((args.reps, args.epochs))
    epoch_train_acc = np.zeros((args.reps, args.epochs))
    epoch_valid_loss = np.zeros((args.reps, args.epochs))
    epoch_valid_acc = np.zeros((args.reps, args.epochs))
    epoch_test_loss = np.zeros((args.reps, args.epochs))
    epoch_test_acc = np.zeros((args.reps, args.epochs))
    saved_model_loss = np.zeros(args.reps)
    saved_model_acc = np.zeros(args.reps)
    
    for r in range(args.reps):
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
        ### define model
        model = GNN(args.num_block,
                    num_features, num_classes, 
                    args.conv_type, args.norm_type, args.conv_hid, args.num_conv_layer, 
                    args.pool_type, args.drop_ratio, args.pRatio, 
                    args.fc_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.filename+'_latest.pth')
        
        ### Training
        print("****** Rep {}: training start******".format(r+1))
        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = loss_criteria(out, data.y, reduction='sum')
                loss.backward()
                optimizer.step()
            if args.dataset == 'qm7':
                train_loss = qm7_test_train(model, train_loader, device)
                val_loss = qm7_test(model, val_loader, device, mean, std)
                test_loss = qm7_test(model, test_loader, device, mean, std)
                print("Epoch {}: Training loss: {:5f}, Validation loss: {:5f}, Test loss: {:.5f}".format(epoch+1, train_loss, val_loss, test_loss))
            else:       
                train_acc, train_loss = test(model, train_loader, device)
                val_acc, val_loss = test(model, val_loader, device)
                test_acc, test_loss = test(model, test_loader, device)
                epoch_train_acc[r, epoch],epoch_valid_acc[r, epoch],epoch_test_acc[r, epoch] = train_acc,val_acc,test_acc
                print("Epoch {}: Training accuracy: {:.5f}; Validation accuracy: {:.5f}; Test accuracy: {:.5f}".format(epoch+1, train_acc, val_acc, test_acc))
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping \n")
                break
            scheduler.step(val_loss)
            
            epoch_train_loss[r, epoch] = train_loss            
            epoch_valid_loss[r, epoch] = val_loss            
            epoch_test_loss[r, epoch] = test_loss

        ### Test
        print("****** Test start ******")
        model.load_state_dict(torch.load(args.filename+'_latest.pth'))
        if args.dataset == 'qm7':
            test_loss = qm7_test(model, test_loader, device, mean, std)
            print("Test Loss: {:.5f}".format(test_loss))
        else:
            test_acc, test_loss = test(model, test_loader, device)
            saved_model_acc[r] = test_acc
            print("Test accuracy: {:.5f}".format(test_acc))
        saved_model_loss[r] = test_loss

    ### save the results
    np.savez(args.filename+".npz",
             epoch_train_loss=epoch_train_loss,
             epoch_train_acc=epoch_train_acc,
             epoch_valid_loss=epoch_valid_loss,
             epoch_valid_acc=epoch_valid_acc,
             epoch_test_loss=epoch_test_loss,
             epoch_test_acc=epoch_test_acc,
             saved_model_loss=saved_model_loss,
             saved_model_acc=saved_model_acc)  
    
#%%
if __name__ == "__main__":
    main()
    
