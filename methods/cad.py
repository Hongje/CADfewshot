
import backbone
import torch, math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate



class MAB(nn.Module):
    '''
        Modified from SetTransformer [Lee et al. 2019, ICML]
        https://github.com/juho-lee/set_transformer
    '''
    def __init__(self, num_heads, dim_Q, dim_K, dim_V, ln=False):
        super(MAB, self).__init__()
        self.dim_V     = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, support=False):
        
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
    
        dim_split = self.dim_V // self.num_heads

        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2) # 1 for better values
        A = A.mean(1,keepdim=True)

        if support:# attention pooled prototype from n_support -> 1 center embedding
            O = A.bmm(V_).squeeze(1)
        else:
            O = torch.cat((K_ + A.bmm(V_)).split(K.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        return O, A

class CAD(MetaTemplate):
    def __init__(self, params, model_func,  n_way, n_support):
        super(CAD, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        self.mab = MAB(params.n_heads, self.feat_dim,self.feat_dim, self.feat_dim, ln=False)
        self.cos_dist = params.cos_dist
        print(f'HEADS : {params.n_heads} | cos_dist {self.cos_dist}')
 

    def set_forward(self, x, is_feature = False):
        ''' Using All Branches in CAD  '''
        # Proposed.

        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_support   = z_support.view(self.n_way, self.n_support, -1 )#the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous()
        z_query     = z_query.view(self.n_way, self.n_query, -1 )

        # MAB (Q,S) Adaptation
        if self.n_support == 1:
            z_proto, _ = self.mab(z_query,z_support,True)
        else:
            z_proto, _ = self.mab(z_query,z_support)
            z_proto    = z_proto.mean(1)

        # MAB (S,Q) Adaptation
        z_query, _ = self.mab(z_support,z_query)
        
        z_query = z_query.view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
    
    def set_forward_support(self,x,is_feature = False):
        ''' Using Support MAB Branch Only  '''
        # No Adaptation of the Queries
        # Ablation Setting 1. cad_sup_mab_only / config
        
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_support   = z_support.view(self.n_way, self.n_support, -1 )#the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous()
        z_query     = z_query.view(self.n_way, self.n_query, -1 )

        # MAB (Q,S) Only
        z_proto, _ = self.mab(z_query,z_support)
        z_proto    = z_proto.mean(1)

        # MAB (Q,Q) No Query Attention
        z_query, _ = self.mab(z_query,z_query)
            
        z_query = z_query.view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_query(self, x, is_feature = False):
        ''' Using Query MAB Branch Only  '''
        # No Adaptation of the Queries
        # Ablation Setting 2. cad_query_mab_only / config
        # 
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_support   = z_support.view(self.n_way, self.n_support, -1 )#the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous()
        z_query     = z_query.view(self.n_way, self.n_query, -1 )

        # MAB (S,S) No Query Attention used.
        if self.n_support == 1:
            z_proto, _ = self.mab(z_support,z_support,True)
        else:
            z_proto, _ = self.mab(z_support,z_support)
            z_proto    = z_proto.mean(1)

        # MAB (S,Q) Query Adaptation Only
        z_query, _ = self.mab(z_support,z_query)
            
        z_query = z_query.view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
    
    def set_forward_supportquery(self, x, is_feature = False):
        # supportquery
        ''' Using Self-Att in both Query/Support '''
        # No Adaptation of the Queries/Support
        # Ablation Setting 2. cad_querysup_self_only / config
        # 

        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_support   = z_support.view(self.n_way, self.n_support, -1 )#the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous()
        z_query     = z_query.view(self.n_way, self.n_query, -1 )

        # MAB (S,S) No Query Adaptation
        z_proto, _ = self.mab(z_support,z_support)
        z_proto    = z_proto.mean(1)

        # MAB (Q,Q) No Support Attention 
        z_query, _ = self.mab(z_query,z_query)
            
        z_query = z_query.view(self.n_way* self.n_query, -1 )

        dists  = euclidean_dist(z_query, z_proto)
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