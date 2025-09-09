import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
"""
    PART of the code is from the following link
    https://github.com/Diego999/pyGAT/blob/master/layers.py
"""


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class IWPA(nn.Module):
    """
    Part attention layer, "Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification"
    """
    def __init__(self, in_channels, part = 3, fuse='sum', inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()
        
        self.fuse = fuse
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.l2norm = Normalize(2)

        if self.inter_channels is None:
            self.inter_channels = in_channels

        if self.out_channels is None:
            self.out_channels = in_channels

        conv_nd = nn.Conv2d

        self.fc1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.fc2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.fc3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)


        self.bottleneck = nn.BatchNorm1d(in_channels) if fuse == 'sum' else nn.BatchNorm1d(in_channels*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

        # weighting vector of the part features
        self.gate = nn.Parameter(torch.FloatTensor(part))
        nn.init.constant_(self.gate, 1/part)
    def forward(self, x, feat, t=None, part=0):
        bt, c, h, w = x.shape
        b = bt // t

        # get part features
        part_feat = F.adaptive_avg_pool2d(x, (part, 1))
        part_feat = part_feat.view(b, t, c, part)
        part_feat = part_feat.permute(0, 2, 1, 3) # B, C, T, Part

        part_feat1 = self.fc1(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat1 = part_feat1.permute(0, 2, 1)  # B, T*Part, C//r

        part_feat2 = self.fc2(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part

        part_feat3 = self.fc3(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat3 = part_feat3.permute(0, 2, 1)   # B, T*Part, C//r

        # get cross-part attention
        cpa_att = torch.matmul(part_feat1, part_feat2) # B, T*Part, T*Part
        cpa_att = F.softmax(cpa_att, dim=-1)

        # collect contextual information
        refined_part_feat = torch.matmul(cpa_att, part_feat3) # B, T*Part, C//r
        refined_part_feat = refined_part_feat.permute(0, 2, 1).contiguous() # B, C//r, T*Part
        refined_part_feat = refined_part_feat.view(b, self.inter_channels, part) # B, C//r, T, Part

        gate = F.softmax(self.gate, dim=-1)
        weight_part_feat = torch.matmul(refined_part_feat, gate)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # weight_part_feat = weight_part_feat + x.view(x.size(0), x.size(1))

        if self.fuse == 'sum':
            feats = weight_part_feat + feat
        elif self.fuse == 'cat':
            feats = torch.cat([weight_part_feat, feat], dim=1)
        feats_bn = self.bottleneck(feats)

        return feats, feats_bn
    
class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, d_k, d_v, nheads, attn_drop=0.1, resid_drop=0.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param nheads: Number of heads
        '''
        super(CrossAttentionModule, self).__init__()
        assert d_k % nheads == 0 and d_v % nheads == 0
        self.d_mode = d_model
        self.d_k = d_model // nheads
        self.d_q = self.d_k
        self.d_v = d_model // nheads
        self.nheads = nheads

        # key, query, value projections for all heads
        # for rgb
        self.rgb_q_proj = nn.Linear(d_model, nheads * self.d_q)
        self.rgb_k_proj = nn.Linear(d_model, nheads * self.d_k)
        self.rgb_v_proj = nn.Linear(d_model, nheads * self.d_v)
        # for ir
        self.ir_q_proj = nn.Linear(d_model, nheads * self.d_q)
        self.ir_k_proj = nn.Linear(d_model, nheads * self.d_k)
        self.ir_v_proj = nn.Linear(d_model, nheads * self.d_v)
        # output projection
        self.rgb_out_proj = nn.Linear(nheads * self.d_v, d_model)
        self.ir_out_proj = nn.Linear(nheads * self.d_v, d_model)

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, rgb_x, ir_x, attn_mask=None):
        """
        Compute cross-attention for rgb and ir 
        Args:
            rgb_x: RGB feature map with shape (B, C, H, W)
            ir_x: IR feature map with shape (B, C, H, W)
            attn_mask: mask for attention with shape (B, nheads, H, W)
        Returns:
            rgb_x: RGB feature map with shape (B, C, H, W)
            ir_x: IR feature map with shape (B, C, H, W)
        """
        b, c, h, w = rgb_x.shape
        rgb_fea_flat = rearrange(rgb_x, 'b c h w -> b (h w) c', h=h, w=w)
        ir_fea_flat = rearrange(ir_x, 'b c h w -> b (h w) c', h=h, w=w)       
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.ln1(rgb_fea_flat)
        q_vis = self.rgb_q_proj(rgb_fea_flat).contiguous().view(b_s, nq, self.nheads, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.rgb_k_proj(rgb_fea_flat).contiguous().view(b_s, nk, self.nheads, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.rgb_v_proj(rgb_fea_flat).contiguous().view(b_s, nk, self.nheads, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.ln2(ir_fea_flat)
        q_ir = self.ir_q_proj(ir_fea_flat).contiguous().view(b_s, nq, self.nheads, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.ir_k_proj(ir_fea_flat).contiguous().view(b_s, nk, self.nheads, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.ir_v_proj(ir_fea_flat).contiguous().view(b_s, nk, self.nheads, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.nheads * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.rgb_out_proj(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.nheads * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.ir_out_proj(out_ir)) # (b_s, nq, d_model)

        out_vis = rearrange(out_vis, 'b (h w) c->b c h w', h=h, w=w)
        out_ir = rearrange(out_ir, 'b (h w) c->b c h w', h=h, w=w)

        return out_vis, out_ir
    
if __name__ == '__main__':
    cross_attn = CrossAttentionModule(d_model=2048, d_k=2048, d_v=2048, nheads=8)
    input_tensor1 = torch.rand(64, 2048, 8, 16)
    input_tensor2 = torch.rand(64, 2048, 8, 16)

    output_tensor1, output_tensor2 = cross_attn(input_tensor1, input_tensor2)
    print(output_tensor1.shape, output_tensor2.shape)