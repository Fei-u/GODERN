import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F

class GODE(nn.Module):
    def __init__(self, func, batch_size, num_node, hid_dim, is_encoder, is_ode, is_aug, is_missing, device):
        super(GODE, self).__init__()
        
        self.batch_size = batch_size
        self.num_node = num_node
        self.hid_dim = hid_dim
        self.func1, self.func2 = func
        self.encoder = is_encoder
        self.is_aug = is_aug
        self.is_ode = is_ode
        self.is_missing = is_missing
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(2*hid_dim, 2*hid_dim),
            nn.ReLU(),
            nn.Linear(2*hid_dim, hid_dim),
            nn.Tanh()
        )     
        
        self.eps = nn.Parameter(torch.zeros(batch_size, num_node, int(hid_dim/2)))
        
        if self.encoder:
            self.solver='euler'
        else:
            self.solver='dopri5'

    def forward(self, input, hz):

        h = self.mlp(torch.cat([input,hz], dim=-1))
        interval = torch.tensor([0., 0.01]).to(self.device)
        for i in range(2):
            h_f = torchdiffeq.odeint_adjoint(func=self.func1,
                                y0=h,
                                t=interval,
                                method=self.solver)
            h_f = h_f[-1,:,:,:int(self.hid_dim/2)]
            
            h = torch.cat([h_f,h[:,:,int(self.hid_dim/2):]], dim=-1)
            h_s = torchdiffeq.odeint_adjoint(func=self.func2,
                                    y0=h,
                                    t=interval,
                                    method=self.solver)
            eps = torch.clamp(self.eps, 0, 0.1)
            h_s = eps*h_s[-1,:,:,int(self.hid_dim/2):]
            h = torch.cat([h_f,h_s], dim=-1)

        return h
    
    
class gode_rnn_cell(nn.Module):
    def __init__(self, args, func, is_encoder):
        super(gode_rnn_cell, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.is_ode = args.is_ode
        self.gode = GODE(func, args.batch_size, args.num_nodes, args.hid_dim, is_encoder, args.is_ode, args.is_aug, args.missing_test, args.device)
        
    def forward(self, inputs, hx):
        hx = self.gode(inputs, hx)
        return hx


class gode_rnn(nn.Module):
    def __init__(self, args, func, num_rnn_layers, is_encoder):
        super(gode_rnn, self).__init__()
    
        self.args = args
        self.num_rnn_layers = num_rnn_layers
        self.agcrnn_layers = nn.ModuleList(
            [gode_rnn_cell(args, func, is_encoder) for _ in range(self.num_rnn_layers)])
        
    def forward(self, inputs, hidden_state):
        hidden_states = []
        output = inputs
        batch, nodes, dim = output.shape[0],output.shape[1],output.shape[2]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch, nodes, dim),
                                       device=self.args.device)
            
        for layer_num, agcrnn_layer in enumerate(self.agcrnn_layers):
            next_hidden_state = agcrnn_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        
        return output, torch.stack(hidden_states)