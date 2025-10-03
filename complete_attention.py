
import torch.nn as nn
import torch



class MultiHeadAttention(nn.Module):
    def __init__(self,i_dim,o_im,context_window,head,dropout=0,bias=False):
        super().__init__()
        assert o_im % head==0  ,"head can not divided "
        self.head=head
        self.out_dim=o_im
        self.head_dim= o_im // head  
        self.droout=nn.Dropout(dropout)
        self.w_q=nn.Linear(i_dim,o_im,bias=bias)
        self.w_k=nn.Linear(i_dim,o_im,bias=bias)
        self.w_v=nn.Linear(i_dim,o_im,bias=bias)
        self.register_buffer("mask",torch.triu(torch.ones(context_window,context_window),diagonal=1))
        self.linear_pro=nn.Linear(o_im,o_im)

    def forward(self,x):
        batch,number_tokens,out_dim=x.shape
        v_q=self.w_q(x) #linear_project 
        v_k=self.w_k(x) #linear_project
        v_v=self.w_v(x) #linear_project

        #Splite the weights --->(batch,number_token,out_dim)--->(batch,number_token,no_head,head_dim)
        Query=v_q.view(batch,number_tokens,self.head,self.head_dim)
        key =v_k.view(batch,number_tokens,self.head,self.head_dim)
        value=v_v.view(batch,number_tokens,self.head,self.head_dim) 

        #change the dimension values number_tokens --self.head

        Query=Query.transpose(1,2)
        key=key.transpose(1,2)      
        value=value.transpose(1,2) 

        Attentions_score= Query @ key.transpose(2,3)
        Attentions_score.masked_fill(self.mask.bool()[:number_tokens,:number_tokens],-torch.inf)
        Attentions_weights=torch.softmax(Attentions_score / key.shape[-1]**0.5,dim=-1)
        Attentions_weights=self.droout(Attentions_weights)
        content=(Attentions_weights @ value).transpose(1,2)
        context=content.contiguous().view(batch,number_tokens,self.out_dim)
        contextual=self.linear_pro(context)

        return contextual 
        
        



class LayerNorm(nn.Module):
    def __init__(self,emd_dim):
        super().__init__()
        self.help=3e-5
        self.scale=nn.Parameter(torch.ones(emd_dim))
        self.shift=nn.Parameter(torch.zeros(emd_dim))
    def forward(self, x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        out_norm=(x-mean)/torch.sqrt(var+self.help)
        return self.scale * out_norm + self.shift 
    

class FFN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]) 
            )
        
    def forward(self,x):
        return self.layers(x)
      




        