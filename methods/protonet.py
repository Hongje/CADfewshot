
import torch, math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_mab(self,x,is_feature = False):
        '''
            ProtoNet with non-parametric MAB
        '''
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_support   = z_support.view(self.n_way, self.n_support, -1 )#the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous()
        z_query     = z_query.view(self.n_way, self.n_query, -1 )

        AS = torch.softmax(z_query.bmm(z_support.transpose(1, 2)) / math.sqrt(self.feat_dim), 2) # 1 for better values
        AS = AS.mean(1,keepdim=True)
        
        z_proto = torch.cat((z_support + AS.bmm(z_support)).split(z_support.size(0), 0), 2)
        z_proto = z_proto.mean(1)

        AQ = torch.softmax(z_support.bmm(z_query.transpose(1, 2)) / math.sqrt(self.feat_dim), 2) # 1 for better values
        AQ = AQ.mean(1,keepdim=True)
        z_query = torch.cat((z_query + AQ.bmm(z_query)).split(z_query.size(0), 0), 2)
        z_query = z_query.view(self.n_way * self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)