import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch import nn as nn
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=3, help='batch size')##512
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='ENZYMES', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
###
from typing import Optional, Callable, List
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data

#from typing import Optional, Callable, List
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data

###假设原数据被以tensor形式保存为.npy格式
##tensor to txt:
##import scipy.io as io
##result1 = np.array(result1)
##np.savetxt('npresult1.txt',result1)

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data

def read_myds_data(i):
    data=[]
    feats = torch.load('/netp/test_1/' + str(i)+"_"+'my_file.npy' )
    j = 0

    feats_n = torch.load('/netp/test_1_/' + str(i) + "_"+'my_file.npy')
    graphs = []
    list_1 = []
    list_2 = []

    list_3 = []
    list_4 = []
    x = torch.ones([32 ,128], dtype=torch.float)
    h = 33##h:number for nodes ;32 for node of conv1.lin_rel.bias while 
        ##0-31 for extracted features for classify and 33 for the first node for conv1.lin_rel.weight 
    h_r = []
   ## h_1 = 3*feats['conv1.lin_rel.weight'].shape[1]##h_1 for edges
    ###

    x = feats['net']['conv1.lin.weight'].t().type(torch.long)
    #x = x.cuda
    print(feats['net']['conv1.lin.weight'].t().shape[0])
    for i_1 in range(0, feats['net']['conv1.lin.weight'].t().shape[0]):
        ##for here feats[name_0].shape[1] would always be the same as feats[name_2].shape[1] I deal with this 2 in one for loop
        list_1.append(h + i_1)
        list_2.append(h - 1)
    edge_list = torch.Tensor([list_1, list_2]).type(torch.long)
    ##edge_list_ = torch.Tensor([list_1, list_2]).type(torch.long)
    
    batch = torch.zeros(12)
    if i == 1:
        batch = torch.zeros(feats['net']['conv1.lin.weight'].t().shape[0])
    else:
        batch_new = (i - 1)*torch.ones(feats['net']['conv1.lin.weight'].t().shape[0])
        batch = torch.cat((batch, batch_new), dim = 0).type(torch.long)##


    ###
    feats_n = torch.load('/netp/test_1_/' + str(i) + "_"+'my_file.npy')##
    y = 1#feats['net']['conv1.lin.weight'].t()
    data=Data(x=x,edge_index=edge_list,y = y) #包装成Data类
    #self.data, self.slices = split(data, batch)
    ##self.data, self.slices = self.collate(data)
    return data##Data(x=x,edge_index=edge_list,y = i)##, slices



###
dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
##dataset = Myds("/netp", "try_")##
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)
###
data_list = []
i_ = 0
'''for i in range(3,25):
    ##int(len(data_list) * 0.8)-1):
    i_ = i_+1
    ###
    data=[]
    feats = torch.load('/netp/test__1/' + str(i)+'my_file.npy' )
    j = 0

    feats_n = torch.load('/netp/test__1_/' + str(i) +'my_file.npy')
    graphs = []
    list_1 = []
    list_2 = []

    list_3 = []
    list_4 = []
    x = torch.ones([128 ,4], dtype=torch.float)
    h = 33##h:number for nodes ;32 for node of conv1.lin_rel.bias while 
        ##0-31 for extracted features for classify and 33 for the first node for conv1.lin_rel.weight 
    h_r = []
   ## h_1 = 3*feats['conv1.lin_rel.weight'].shape[1]##h_1 for edges
    ###

    x = feats['net']['conv1.lin.weight'].type(torch.long)
    #x = x.cuda
    #print(feats['net']['conv1.lin.weight'].t().shape[0])
    for i_1 in range(1, feats['net']['conv1.lin.weight'].t().shape[0]):
        ##for here feats[name_0].shape[1] would always be the same as feats[name_2].shape[1] I deal with this 2 in one for loop
        list_1.append(h + i_1)
        list_2.append(h - 1)
    edge_list = torch.Tensor([list_1, list_2]).type(torch.long)
    ##edge_list_ = torch.Tensor([list_1, list_2]).type(torch.long)
    
    batch = torch.zeros(12)
    if i == 1:
        batch = torch.zeros(feats['net']['conv1.lin.weight'].shape[0])
    else:
        batch_new = (i - 1)*torch.ones(feats['net']['conv1.lin.weight'].t().shape[0])
        batch = torch.cat((batch, batch_new), dim = 0).type(torch.long)##



    ##feats_n = torch.load('/netp/test__1_/' + str(i) + "_"+'my_file.npy')##
    y =1## feats_n['net']['conv1.lin.weight'].type(torch.float)
    ###
    #print('1:',dataset[i].x.shape)

    #print('2:',dataset[i].edge_index.shape)
    #print('3:',dataset[i].y.type)
    dataset[i].x = x##read_myds_data(i).x#torch.ones(dataset[i].x.shape).type(torch.long)
    dataset[i].edge_index = edge_list##read_myds_data(i).edge_index##torch.ones(dataset[i].edge_index.shape).type(torch.long)
    dataset[i].y = y-x
    
    data_list.append(dataset[i])#调用函数读取数据，包装成Data'''

###
num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


model = Model(args).to(args.device)
##model_1 = Model(args).to(args.device)##
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    i_1 = 0
    i_count = 0####
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        #self.nhid, self.num_classes####
        #radt = None, radb = None, radidx = None, ridxuder5, label = None)
        h = 0
        
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            data = data.to(args.device)
            ###count radidx here:
            n_1 = torch.nonzero(F.relu(torch.ones(data.batch.shape).cuda() - data.batch) ).shape[0]
            n_2_ = torch.nonzero(F.relu(2 * torch.ones(data.batch.shape).cuda() - data.batch) ).shape[0]
            n_2 = n_2_ - n_1
            n_3_ = data.batch.shape[0] 
            n_3 = n_3_ - n_2_
            ridxuder5 = False
            if n_1 < 10:
                #print('wrong')
                ridxuder5 = True
            if n_2 < 10:
                #print('wrong')
                ridxuder5 = True
            if n_3 < 10:
                ridxuder5 = True
                #print('wrong')
            if ridxuder5 == False:
                radidx_0 = torch.reshape(torch.randint(0, n_1 - 1, (5, ) ), [1,5])
                radidx_1 = torch.reshape(torch.randint(n_1, n_2_ - 1, (5, ) ), [1,5])
                radidx_2 = torch.reshape(torch.randint(n_2_, n_3_ - 1, (5, ) ), [1,5])
                radidx = torch.cat((radidx_0, radidx_1, radidx_2), dim = 0)
                ###count radt and radb here
                radt = torch.randn(128)
                radb = torch.randn(128)            
                ###count label here
                label = torch.randint(0,3,(3,)).cuda()
                #print(label[0])
                ###
            
                ##print(data.validate())
                out_1 = model(data,radt = radt, radb = radb, radidx = radidx, label = label)
                out = model(data)
                out_2 = model(data, if_ = 0.75)
                #, rad = None, radidx = None, label = None
                #with torch.no_grad():
                 #   out_1 = out_
            else:
                h = h+1
                out_1 = model(data, if_ = 0.10)
                out = model(data)
                out_2 = model(data, if_ = 0.75)   
                
                    
                
            
            #loss = loss_.cuda()
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            loss = F.nll_loss(out, data.y) + triplet_loss(out_1, out, out_2) #+ 0.5*F.nll_loss(out_1, data.y)##,weight = loss_)
            loss.backward()
            '''
            if i == 1:
                print("data.y:",out)
                print("x_s:",out.shape)
                print("data.y[2]:",data.y[2])
            
            for item in model.state_dict():
                print(item)
                print('names:',str(item))
                print('parasize:',model.state_dict()[str(item)].shape)
                
            ###
            
            a = torch.randn(1)
            str(i_1)
            if a < 0:
                dict_1 = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                store = '/netp/test__1/' + str(i_1)+'my_file.npy'          
                torch.save(dict_1, store)
                #dict_g = {'g_g': g_g, 'g_e': g_e}
                ##torch.save(dict_g, store)

            
            i_count = i_count + 1
            if i_count == 3:
                i_count = 0
                  
                dict_2 = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                store = '/netp/test__1_/' + str(i_1)+'my_file.npy'          
                torch.save(dict_1, store)
                loss = loss.cpu()
                dict_3 = {'loss':loss}
                torch.save(loss, store)
                i_1 = i_1 + 1 
            '''
            
            
            optimizer.step()
            ###
            
            
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        print("i:",i)####
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / (10+len(loader.dataset)), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    '''for param in model.parameters():
        print("name:", param.name)
        print('size:', param.shape)'''
    
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))

