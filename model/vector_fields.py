import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.output_channels = self.hidden_channels
            
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, self.output_channels) 


    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
            
        z = self.linear_out(z)
        z = z.tanh()
        return z


class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim, is_ode):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.output_channels = self.hidden_channels
        if not is_ode:
            self.hidden_channels = hidden_channels-10
            self.output_channels = hidden_channels-10
        self.linear_in = torch.nn.Linear(self.hidden_channels, hidden_hidden_channels)
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, self.output_channels)
        self.cheb_k = cheb_k
        self.weights_pool1 = nn.Parameter(torch.zeros(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
        self.weights_pool2 = nn.Parameter(torch.zeros(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
        self.bias_pool1 = nn.Parameter(torch.zeros(embed_dim, hidden_hidden_channels))
        self.bias_pool2 = nn.Parameter(torch.zeros(embed_dim, hidden_hidden_channels))



    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()
        z = self.agc(z)
        z = z.tanh()
        z = self.linear_out(z)
        z = z.tanh()
        
        return z 
    
    def set_nodevec(self, nodevec_1, nodevec_2):
        self.nodevec_1 = nodevec_1
        self.nodevec_2 = nodevec_2

    def agc(self, z):

        node_num = self.nodevec_1.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.nodevec_1, self.nodevec_2.transpose(1,0))-torch.mm(self.nodevec_2, self.nodevec_1.transpose(1,0))),dim=1)
        laplacian=False
        if laplacian == True:
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights_1 = torch.einsum('nd,dkio->nkio', self.nodevec_1, self.weights_pool1)  
        weights_2 = torch.einsum('nd,dkio->nkio', self.nodevec_2, self.weights_pool2)  
        bias_1 = torch.matmul(self.nodevec_1, self.bias_pool1)                       
        bias_2 = torch.matmul(self.nodevec_2, self.bias_pool2)  
        x_g = torch.einsum("knm,bmc->bknc", supports, z)     
        x_g = x_g.permute(0, 2, 1, 3) 
        z = torch.einsum('bnki,nkio->bno', x_g, weights_1-weights_2) + (bias_1-bias_2)    
        return z





    