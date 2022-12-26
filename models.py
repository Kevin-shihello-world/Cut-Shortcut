import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from layers import GCN, HGPSLPool
from torch.nn import Linear, BatchNorm1d##


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.bn = BatchNorm1d(self.nhid)##

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data, if_ = None, radt = None, radb = None, radidx = None, label = None):
        ##num of radidx = 5
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x__ = x
        if radt != None:
            var = torch.var(x).cuda()
            #print("label:", label)
            '''
            radidx_0 = torch.reshape(torch.randint(0, n_1 - 1, (5, ) ), [1,5])
            radidx_1 = torch.reshape(torch.randint(n_1, n_2_ - 1, (5, ) ), [1,5])
            radidx_2 = torch.reshape(torch.randint(n_2_, n_3_ - 1, (5, ) ), [1,5])
            radidx = torch.cat((radidx_0, radidx_1, radidx_2), dim = 0)'''
            Zeros = torch.zeros(x.shape).cuda()
            for i in range(0, radidx.shape[0]):
                radidx = radidx.cuda()
                radidx_ = torch.reshape(torch.randint(int(radidx[i][2]), int(radidx[i][2] + radidx[i][1]), (9, ) ), [1,9])
                radt = var*torch.reshape(radt, [1,128]).cuda()
                radb = radb.cuda()
                
            #label = torch.reshape(label, [1,3]).float()
                label = label.cuda()
                test = radidx[i][0].item()
                a = int(test)
                b = label[a]
                label_ = torch.reshape(b, [1]).float()
                Label_ = F.linear(label_, radt.t(), bias=radb)
                Zeros[radidx_[i][0]] = 0.25*torch.reshape(Label_, [1,128])
            #label_ = torch.reshape(label[1], [1]).float()
            #label_2_ = torch.reshape(label[2], [1]).float()
            #print('label0:', label.shape)
            #print('zeros:',Zeros.shape)
            
            #print('new:', radidx[1][4])
            #Label_1 = F.linear(label_1_, radt.t(), bias=radb)
            #Label_2 = F.linear(label_2_, radt.t(), bias=radb)
            '''for i in range(0,5):
                Zeros[radidx[0][i]] = 0.5*torch.reshape(Label_0, [1,128])
            for i in range(0,5):
                Zeros[radidx[1][i]] = 0.5*torch.reshape(Label_1, [1,128])
            for i in range(0,5):
                Zeros[radidx[2][i]] = 0.5*torch.reshape(Label_2, [1,128])'''
            
            label_ = Zeros
            x = x + label_
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.bn(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        
        x = F.relu(self.lin1(x))
        '''if if_ != None:
            x = F.dropout(x, p = if_, training=self.training)'''
        
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x_ = x
        x = F.log_softmax(self.lin3(x), dim=-1)
        if if_ != None:
            return x_, x__
        return x_, x, x__
