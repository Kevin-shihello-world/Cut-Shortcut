import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch import nn as nn
from models import Model
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')##512
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



###
dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
##dataset = Myds("/netp", "try_")##
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

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
    model.load_state_dict(torch.load('/HGP-SL/test__/1.pth'))
    model.train()
    i_1 = 0
    i_count = 0####
    #cos_ = F.cosine_similarity(dim=0)##, eps=1e-6)
    #torch.save(model.state_dict(),'HGP-SL/modelinicial/1.pth')
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        #self.nhid, self.num_classes####
        #radt = None, radb = None, radidx = None, ridxuder5, label = None)
        h = 0
        
        for i, data in enumerate(train_loader):
            batch_num = data.y.shape[0]
            optimizer.zero_grad()
            a = torch.randn(1)
            data = data.to(args.device)
            radidx = torch.reshape(torch.ones(batch_num), [1,batch_num])
            
            for j in range(0, data.y.shape[0]):
                if radidx.shape[0] == 1:
                    radidx_ = torch.count_nonzero(F.relu(j*torch.ones(data.batch.shape).cuda() - data.batch) )


                else:
                    radidx__ = torch.count_nonzero(F.relu(j*torch.ones(data.batch.shape).cuda() - data.batch) )
                    #item_ = radidx.shape[0] - 1
                    radidx_ = radidx__ - max #radidx[item_][0].item()
                    begin = max
                    max = radidx__
                if radidx_ > 100: #
                    a = a+1
                    if radidx.shape[0] == 1:
                        ###val = [float(i), radidx_]
                        begin = 0
                        radidx = torch.reshape(torch.Tensor([float(j), radidx_, begin]), [1,3])
                        max = radidx_

                    else:
                        radidx_ = torch.reshape(torch.Tensor([float(j), radidx_, begin]), [1,3])
                        radidx = torch.cat((radidx, radidx_), dim = 0)
                    #j = j+1
            ###
            
            ###count radidx here:
            '''n_1 = torch.nonzero(F.relu(torch.ones(data.batch.shape).cuda() - data.batch) ).shape[0]
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
            #ridxuder5 = True
            if ridxuder5 == False:
                radidx_0 = torch.reshape(torch.randint(0, n_1 - 1, (5, ) ), [1,5])
                radidx_1 = torch.reshape(torch.randint(n_1, n_2_ - 1, (5, ) ), [1,5])
                radidx_2 = torch.reshape(torch.randint(n_2_, n_3_ - 1, (5, ) ), [1,5])
                radidx = torch.cat((radidx_0, radidx_1, radidx_2), dim = 0)
                ###count radt and radb here
                radt = torch.randn(batch_num)
                radb = torch.randn(batch_num)            
                ###count label here
                label = torch.randint(0,3,(3,)).cuda()
                #print(label[0])'''
            
                ###
            
                ##print(data.validate())
                
            radt = torch.randn(128)
            radb = torch.randn(128)  
            label = data.y
            if epoch < 2: 
                out_,out, out_t = model(data)
                #with torch.no_grad():
                 #   out_ = out

                loss = F.nll_loss(out, data.y)#- cos_

            else:
                #if epoch > 50:
                out_1, out_1_ = model(data, if_ = 1,radt = radt, radb = radb, radidx = radidx, label = label)
                r_length = data.y.shape[0]
                radm = torch.randint(0,5,(r_length,)).cuda()
                out_2 = model(data, if_ = 1,radt = radt, radb = radb, radidx = radidx, label = radm)[1]
                out_, out, out_t = model(data)
                with torch.no_grad():
                    out__ = out_
                    out_t_ = out_t
                cosneg = F.cosine_similarity(out_1_.view(-1), out_t_.view(-1), dim = 0)
                cosadd = F.cosine_similarity(out_1.view(-1), out__.view(-1), dim = 0)
                loss = F.nll_loss(out, data.y) - cosneg + cosadd
                
                    
                
                                    
                
                、
                

               
            
            
            loss.backward()
           
            
            optimizer.step()
            ###
            
            
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
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
        #print('data.y:', data.y)
        data = data.to(args.device)
        out_, out, _ = model(data)
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


