
from GPT.complete_attention import MultiHeadAttention,LayerNorm,FFN

import torch.nn as nn
import torch
class Transformer_block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.multihead=MultiHeadAttention(
            cfg["emb_dim"],cfg["emb_dim"],cfg["context_length"],cfg["n_heads"],cfg["drop_rate"],cfg["qkv_bias"]
        )

        self.ffn=FFN(cfg)
        self.layerNorm1=LayerNorm(cfg["emb_dim"])
        self.layerNorm2=LayerNorm(cfg["emb_dim"])
        self.dropout=nn.Dropout(cfg["drop_rate"])


    def forward(self,x):
        
        shortcut=x
        layern=self.layerNorm1(x)
        attent=self.multihead(layern)
        droput=self.dropout(attent)
        x=droput+shortcut

        shortcut2=x

        layern2=self.layerNorm2(x)
        ff=self.ffn(layern2)
        droput2=self.dropout(ff)
        
        context=shortcut2 + droput2

        return context 

    



