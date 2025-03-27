import torch
from torch import nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from models.sphereface import sphere36
import os
from collections import OrderedDict

class TranSSL(nn.Module):
    def __init__(self, out_dim = 512, weights_path = "weights/sphere36_mcp.pth", dropout=0.2, num_heads=8):
        super(TranSSL, self).__init__()
        self.out_dim = out_dim
        self.weights_path = weights_path
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_tokens = [64, 16, 8, 4]
        self.filters = [64, 128, 256, 512]

        #Using sphere36 as backbone
        self.base = sphere36(out_dim)
        self.base.load_state_dict(torch.load(os.path.join(os.getcwd(), weights_path), weights_only=True))
        
        #Different pos enc for different intermediate outputs of model
        self.positional_encodings = nn.ModuleList([
            Summer(PositionalEncoding2D(dim)) for dim in self.filters])
        
        #Flatten the (batch_size, channels, height, width) to (batch_size, channels, height * width)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        
        #Extra layer to prepare transformer input
        self.embedding_transforms = nn.ModuleList([nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=size, out_channels=size, kernel_size=1, bias=False)),
            ('prelu', nn.PReLU())])) for size in self.filters])
        
        #Transformers
        self.mhas = nn.ModuleList([nn.MultiheadAttention(size, num_heads, dropout = dropout, batch_first=True) for size in self.filters])
        
        #Feed forward for output from transformers
        self.ffs = nn.ModuleList([nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(size * num_tok, size * num_tok // 2, bias=False)),
            ('prelu', nn.PReLU()),
            ('linear2', nn.Linear(size * num_tok // 2, out_dim, bias=False)),
            ('prelu', nn.PReLU()),
            ('layernorm',nn.LayerNorm(out_dim))
        ])) for size, num_tok in zip(self.filters, self.num_tokens)])

        #Linears for Filter-based tokenizer
        self.tokenizers = nn.ModuleList([nn.Linear(in_features=in_size, out_features=out_size, bias=False) for in_size, out_size in zip(self.filters, self.num_tokens)])
        self.softmax = nn.Softmax(dim = -2)
        #Final
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(out_dim * 4, out_dim * 2, bias=False)),
            ('prelu', nn.PReLU()),
            ('linear2', nn.Linear(out_dim * 2, out_dim, bias=False)),
            ('prelu', nn.PReLU()),
            ('layernorm',nn.LayerNorm(out_dim))
        ]))
    
    def forward(self, input):
        #First layer
        output = self.base.layer1(input)
        layer1_output = self.embedding_transforms[0](output)
        layer1_output = self.positional_encodings[0](layer1_output)
        #Permute to (batch_size, num_tokens, emb_size)
        layer1_output = self.flatten(layer1_output).permute([0,2,1])
        #Filter-based tokenizer
        layer1_output = self.softmax(self.tokenizers[0](layer1_output)).transpose(-2, -1) @ layer1_output
        layer1_output, _ = self.mhas[0](layer1_output, layer1_output, layer1_output, need_weights = False)
        layer1_output = self.ffs[0](self.flatten(layer1_output))

        #Second layer
        output = self.base.layer2(output)
        layer2_output = self.embedding_transforms[1](output)
        layer2_output = self.positional_encodings[1](layer2_output)
        #Permute to (batch_size, num_tokens, emb_size)
        layer2_output = self.flatten(layer2_output).permute([0,2,1])
        #Filter-based tokenizer
        layer2_output = self.softmax(self.tokenizers[1](layer2_output)).transpose(-2, -1) @ layer2_output
        layer2_output, _ = self.mhas[1](layer2_output, layer2_output, layer2_output, need_weights = False)
        layer2_output = self.ffs[1](self.flatten(layer2_output))

        #Third layer
        output = self.base.layer3(output)
        layer3_output = self.embedding_transforms[2](output)
        layer3_output = self.positional_encodings[2](layer3_output)
        #Permute to (batch_size, num_tokens, emb_size)
        layer3_output = self.flatten(layer3_output).permute([0,2,1])
        #Filter-based tokenizer
        layer3_output = self.softmax(self.tokenizers[2](layer3_output)).transpose(-2, -1) @ layer3_output
        layer3_output, _ = self.mhas[2](layer3_output, layer3_output, layer3_output, need_weights = False)
        layer3_output = self.ffs[2](self.flatten(layer3_output))


        #Fourth layer
        output = self.base.layer4(output)
        layer4_output = self.embedding_transforms[3](output)
        layer4_output = self.positional_encodings[3](layer4_output)
        #Permute to (batch_size, num_tokens, emb_size)
        layer4_output = self.flatten(layer4_output).permute([0,2,1])
        #Filter-based tokenizer
        layer4_output = self.softmax(self.tokenizers[3](layer4_output)).transpose(-2, -1) @ layer4_output
        layer4_output, _ = self.mhas[3](layer4_output, layer4_output, layer4_output, need_weights = False)
        layer4_output = self.ffs[3](self.flatten(layer4_output))

        output = self.fc(torch.cat((layer1_output, layer2_output, layer3_output, layer4_output), -1))
        return output
