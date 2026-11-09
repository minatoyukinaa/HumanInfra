import torch
import torch.nn as nn
import torch.nn.functional as F

def eager_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor):
    scales = (k.shape[-1]) ** -0.5
    # 1. Q @ K^T [seqlen_q,head_dim] * [head_dim,seqlen_kv]
    q = q * scales
    scores = torch.matmul(q,k.transpose(-1,-2))

    scores += mask 
    # softmax()
    scores = F.softmax(scores,dim=-1)  # [batch,heads,seqlen_q,seqlen_kv]
    o = torch.matmul(scores,v) # v shape is [batch,heads,seqlen_kv,head_dim]
    return o

class GQA(nn.Module):
    def __init__(self, dim=8192,q_heads=64,kv_heads=16,head_dim=128):
        super().__init__()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim,q_heads*head_dim)
        self.k_proj = nn.Linear(dim,kv_heads*head_dim)
        self.v_proj = nn.Linear(dim,kv_heads*head_dim)

        self.o_proj = nn.Linear(q_heads*head_dim,dim)

    def forward(self,x:torch.Tensor):
        batch,_,dim = x.shape
        # x:[batch,seq_len,dim]-> [batch,heads,seq_len,dim]
        q = self.q_proj(x).view(batch,-1,self.q_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch,-1,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch,-1,self.kv_heads,self.head_dim).transpose(1,2)

        # repeat kv
        assert self.q_heads % self.kv_heads == 0
        num_repeats = self.q_heads // self.kv_heads
        k = torch.repeat_interleave(k,num_repeats,dim=1) # 在heads上进行repeat
        v = torch.repeat_interleave(v,num_repeats,dim=1)

        seqlen_q,seqlen_kv = q.shape[-2],k.shape[-2]
        tri = torch.triu(
        torch.ones((seqlen_q,seqlen_kv),dtype=torch.bool,device=q.device)
        ,diagonal=1)
        casual = torch.where(tri,float("-inf"),0).to(q.dtype)

        o = eager_attention(q,k,v,casual).reshape(batch,-1,dim)

        o2 = F.scaled_dot_product_attention(q,k,v,attn_mask=casual).reshape(batch,-1,dim)

        # print(f"cos of :{torch.cosine_similarity(o,o2)}")
        torch.testing.assert_close(o2,o)
        o = self.o_proj(o)
        return o

def test_case():
    device = torch.device("cuda:0")
    model = GQA().to(device,torch.bfloat16)
    batch,seqlen,dim = 8,1024,64*128

    x = torch.randn((batch,seqlen,dim),dtype=torch.bfloat16).cuda()
    y = model(x)


    # 对拍torch.scale_dots
test_case()