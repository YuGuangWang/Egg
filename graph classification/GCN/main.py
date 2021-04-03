import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from mSVD import mSVD
import argparse
import os.path as osp
from QM7 import QM7, qm7_test, qm7_test_train, MyDataset


#%% function for early_stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


#%% GrassPool 
def grasspool(hid, graph_sizes, pRatio):
    """
    cur_node_embeddings: hidden rep for a single graph g_i
    hid: hidden rep of batch_graph to be transformed
    graph_sizes: a list of individual graph node size
    """   
    graph_sizes = graph_sizes.tolist()
    node_embeddings = torch.split(hid, graph_sizes)
    ### create an autograd-able variable
    batch_graphs = torch.zeros(len(graph_sizes), int(hid.shape[1]*(hid.shape[1]+1)/2)).to(hid.device)
       
    for g_i in range(len(graph_sizes)):
        cur_node_embeddings = node_embeddings[g_i]
        U, S, V = mSVD.apply(cur_node_embeddings.t())
        k = sum(S > pRatio).item()
        subspace_sym = torch.matmul(U[:,:k],U[:,:k].t())
        ### flatten
        idx = torch.triu_indices(subspace_sym.shape[0],subspace_sym.shape[0])
        cur_graph_tri_u = subspace_sym[idx[0],idx[1]]
        batch_graphs[g_i] = cur_graph_tri_u.flatten()
    return batch_graphs


#%% FC Layer
def score_block(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(),
                         nn.Dropout(dropout))


#%%    
class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob, hid_fc_dim, pRatio):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.pRatio = pRatio
        self.GConv1 = GCNConv(num_features, nhid)
        self.GConv2 = GCNConv(nhid, nhid)
        
        hidden_dim = [int(nhid*(nhid+1)/2)]+hid_fc_dim+[num_classes]
        fcList = [score_block(i,j,self.dropout_prob) for i,j in zip(hidden_dim[:-2], hidden_dim[1:])]
        fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),
                                    nn.BatchNorm1d(hidden_dim[-1])
                                    ))
        self.fc = nn.Sequential(*fcList) 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        _, graph_size = torch.unique(batch, return_counts = True)
        
        # two convolutional layers
        hidden_rep = F.relu(self.GConv1(x, edge_index))
        hidden_rep = F.relu(self.GConv2(hidden_rep, edge_index))

        # one global pooling layer
        h_pooled= grasspool(hidden_rep, graph_size, self.pRatio)
        x = self.fc(h_pooled)
        if num_classes == 1:
            return x.view(-1)
        else: 
            return x
    

def test(model, loader, device):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.cross_entropy(out, data.y,reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


#%% main block
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping criteria (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--conv_hid', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--num_conv_layer', type=int, default=2,
                        help='number of hidden mlp layers in a conv layer (default: 2)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument("--fc_dim", type=int, nargs="*", default=[64,16],
                        help="dimension of fc hidden layers (default: [64])")
    parser.add_argument('--pRatio', type=float, default=0.8,
     					help='the ratio of info preserved in the Grassmann subspace (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Parameter Setting
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.conv_hid
    epochs = args.epochs
    num_reps = args.reps

    # create results matrix
    epoch_train_loss = np.zeros((num_reps, epochs))
    epoch_train_acc = np.zeros((num_reps, epochs))
    epoch_valid_loss = np.zeros((num_reps, epochs))
    epoch_valid_acc = np.zeros((num_reps, epochs))
    epoch_test_loss = np.zeros((num_reps, epochs))
    epoch_test_acc = np.zeros((num_reps, epochs))
    saved_model_loss = np.zeros(num_reps)
    saved_model_acc = np.zeros(num_reps)

    # training
    for r in range(num_reps):
        training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model = Net(num_features, nhid, num_classes, args.drop_ratio, args.fc_dim, args.pRatio).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.filename+'_latest.pth')

        # start training
        min_loss = 1e10
        patience = 0
        print("****** Rep {}: Training start ******".format(r+1))
        for epoch in range(epochs):
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

        # Test
        print("****** Test start ******")
        model = Net(num_features, nhid, num_classes, args.drop_ratio, args.fc_dim, args.pRatio).to(device)
        model.load_state_dict(torch.load(args.filename+'_latest.pth'))
        if args.dataset == 'qm7':
            test_loss = qm7_test(model, test_loader, device, mean, std)
            print("Test Loss: {:.5f}".format(test_loss))
        else:
            test_acc, test_loss = test(model, test_loader, device)
            saved_model_acc[r] = test_acc
            print("Test accuracy: {:.5f}".format(test_acc))
        saved_model_loss[r] = test_loss

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_train_loss,
             epoch_train_acc=epoch_train_acc,
             epoch_valid_loss=epoch_valid_loss,
             epoch_valid_acc=epoch_valid_acc,
             epoch_test_loss=epoch_test_loss,
             epoch_test_acc=epoch_test_acc,
             saved_model_loss=saved_model_loss,
             saved_model_acc=saved_model_acc)
