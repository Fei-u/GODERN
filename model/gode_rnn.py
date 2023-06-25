import torch
import torch.nn.functional as F
import torch.nn as nn
import torchsde
import torchdiffeq
from vector_fields import *
from gcrnn_cell import gode_rnn

class EncoderGODE(nn.Module):
    def __init__(self, args, func, input_channels, hidden_channels, output_channels, rnn_layers, device):
        super(EncoderGODE, self).__init__()
        self.args = args
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.rnn_layers = rnn_layers
        self.device = device
        
        self.linear_in = torch.nn.Linear(input_channels, hidden_channels)
        
        self.func = func
        
        self.RNN = gode_rnn(args, func, rnn_layers, is_encoder=True)
        
    def forward(self, inputs):
        ## batch lag node dim
        inputs = inputs.transpose(0,1)
        inputs = self.linear_in(inputs)
        ## lag batch node dim
        encoder_hidden_state = None
        for t in range(self.args.lag):
            _, encoder_hidden_state = self.RNN(inputs[t], encoder_hidden_state)
        
        
        return encoder_hidden_state

class DecoderGODE(nn.Module):
    def __init__(self, args, func, input_channels, hidden_channels, output_channels, rnn_layers, device):
        super(DecoderGODE, self).__init__()
        self.args = args
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.rnn_layers = rnn_layers
        self.device = device
        
        self.conv_out = nn.Conv2d(args.horizon, args.horizon, kernel_size=(1, hidden_channels), bias=True).to(device)
        
        self.func = func
        
        self.RNN = gode_rnn(args, func, rnn_layers, is_encoder=True)
            
    def forward(self, encoder_hidden_state):
        batch_size, nodes, = encoder_hidden_state.size(1),encoder_hidden_state.size(2)
        decoder_input = torch.zeros((batch_size, nodes, self.hidden_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        outputs = []

        for t in range(self.args.horizon):
            decoder_output, decoder_hidden_state = self.RNN(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
    
        outputs = torch.stack(outputs)
        outputs = self.conv_out(outputs.transpose(0,1))
        return outputs

 

class Model(nn.Module):
    def __init__(self, args, func, input_channels, hidden_channels, output_channels, rnn_layers, device):
        super().__init__()
        
        self.Encoder = EncoderGODE(args, func, input_channels, hidden_channels, output_channels, rnn_layers, device).to(device) 
        self.Decoder = DecoderGODE(args, func, input_channels, hidden_channels, output_channels, rnn_layers, device).to(device)
        self.num_nodes = args.num_nodes
        self.horizon = args.horizon
        self.func1, self.func2 = func
        
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        
    def forward(self, inputs):
        self.func1.set_nodevec(self.nodevec1, self.nodevec2)
        self.func2.set_nodevec(self.nodevec1, self.nodevec2)        
        h_t = self.Encoder(inputs)
        output = self.Decoder(h_t)

        return output
        
        
        
     
        