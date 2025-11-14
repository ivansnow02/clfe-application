from typing import Tuple

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    DebertaV2Tokenizer,
    DebertaV2Model,
)


TRANSFORMERS_MAP = {
    "bert": (BertModel, BertTokenizer),
    "roberta": (RobertaModel, RobertaTokenizer),
    "deberta": (DebertaV2Model, DebertaV2Tokenizer),
}


class BertTextEncoder(nn.Module):
    def __init__(
        self, use_finetune: bool = False, pretrained: str = "bert-base-uncased"
    ):
        super().__init__()

        model_type = "bert"
        if "roberta" in pretrained.lower():
            model_type = "roberta"
        if "deberta" in pretrained.lower():
            model_type = "deberta"

        tokenizer_class = TRANSFORMERS_MAP[model_type][1]
        model_class = TRANSFORMERS_MAP[model_type][0]

        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune

    def forward(
        self, text: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # text: (B, 3, S) -> ids, mask, seg
        input_ids, input_mask, segment_ids = (
            text[:, 0, :].long(),
            text[:, 1, :].float(),
            text[:, 2, :].long(),
        )

        if self.use_finetune:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                output_hidden_states=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    output_hidden_states=True,
                )

        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states
        return last_hidden_state, all_hidden_states


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, q, k, v):
        b, n, _ = q.shape
        h = self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormAttention(
                    dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                ),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, save_hidden: bool = False):
        if save_hidden:
            hidden_list = [x]
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        token_len,
        save_hidden,
        dim,
        depth,
        heads,
        mlp_dim,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_frames + token_len, dim)
            )
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        if self.token_len is not None:
            extra_token = repeat(self.extra_token, "1 n d -> b n d", b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, : n + self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, L, C)
        x_permuted = x.permute(0, 2, 1)
        b, c, _ = x_permuted.size()
        y = self.avg_pool(x_permuted).view(b, c)
        y = self.fc(y).view(b, c, 1)
        scaled_x = x_permuted * y.expand_as(x_permuted)
        return scaled_x.permute(0, 2, 1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout: float = 0.0):
        super().__init__()
        self.attn = PreNormAttention(
            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        )
        self.ff = PreNormForward(dim, FeedForward(dim, dim * 2, dropout=dropout))

    def forward(self, query, context):
        query = self.attn(query, context, context) + query
        query = self.ff(query) + query
        return query


class GatedGuidanceLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout: float = 0.0):
        super().__init__()
        self.audio_guidance_candidate = CrossAttentionBlock(
            dim, heads, dim_head, dropout
        )
        self.visual_guidance_candidate = CrossAttentionBlock(
            dim, heads, dim_head, dropout
        )
        self.audio_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.visual_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, h_t, h_a_prev, h_v_prev):
        h_a_candidate = self.audio_guidance_candidate(query=h_a_prev, context=h_t)
        h_v_candidate = self.visual_guidance_candidate(query=h_v_prev, context=h_t)
        gate_a = self.audio_gate(torch.cat((h_a_prev, h_a_candidate), dim=-1))
        gate_v = self.visual_gate(torch.cat((h_v_prev, h_v_candidate), dim=-1))
        h_a_new = (1 - gate_a) * h_a_prev + gate_a * h_a_candidate
        h_v_new = (1 - gate_v) * h_v_prev + gate_v * h_v_candidate
        return h_a_new, h_v_new


class HierarchicalGuidanceEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedGuidanceLayer(dim, heads, dim_head, dropout) for _ in range(depth)
        ])
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h_t_list, h_a, h_v):
        for i, layer in enumerate(self.layers):
            h_a, h_v = layer(h_t_list[i], h_a, h_v)
        combined_features = torch.cat((h_a, h_v), dim=-1)
        h_hyper = self.fusion_projection(combined_features)
        return h_hyper


import math


class _MultimodalCrossLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden size ({dim}) not multiple of heads ({heads})")
        self.hidden_size = dim
        self.num_head = heads
        self.attention_head_size = dim // heads
        self.all_head_size = self.num_head * self.attention_head_size
        self.q = nn.Linear(self.hidden_size, self.all_head_size)
        self.k = nn.Linear(self.hidden_size, self.all_head_size)
        self.v = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout_2 = nn.Dropout(dropout)
        self.w_source_param = nn.Parameter(torch.tensor(0.5))
        self.w_target_param = nn.Parameter(torch.tensor(0.5))
        self.gate_source_target_linear = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )
        self.gate_target_source_linear = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.LayerNorm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def transpose1(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def cross_attention(self, q_modal, k_modal, v_modal):
        q_proj, k_proj, v_proj = (
            self.transpose1(self.q(q_modal)),
            self.transpose1(self.k(k_modal)),
            self.transpose1(self.v(v_modal)),
        )
        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )
        attn_prob = nn.Softmax(dim=-1)(attn_score)
        attn_prob = self.dropout_1(attn_prob)
        context = torch.matmul(attn_prob, v_proj).permute(0, 2, 1, 3).contiguous()
        return context.view(context.size(0), context.size(1), -1)

    def forward(self, source_x, target_x):
        s_len, t_len = source_x.shape[1], target_x.shape[1]
        s_t_cross = self.cross_attention(source_x, target_x, target_x)
        t_s_cross = self.cross_attention(target_x, source_x, source_x)
        gate_s = torch.sigmoid(
            self.gate_source_target_linear(torch.cat((source_x, s_t_cross), dim=-1))
        )
        gate_t = torch.sigmoid(
            self.gate_target_source_linear(torch.cat((target_x, t_s_cross), dim=-1))
        )
        h_s_fused = source_x + gate_s * s_t_cross
        h_t_fused = target_x + gate_t * t_s_cross
        fused_hidden = torch.cat([h_s_fused, h_t_fused], dim=1)
        q_layer = self.transpose1(self.q(fused_hidden))
        k_layer = self.transpose1(self.k(fused_hidden))
        v_layer = self.transpose1(self.v(fused_hidden))
        attention_score = torch.matmul(q_layer, k_layer.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )
        attention_prob_s = nn.Softmax(dim=-1)(attention_score[:, :, :, :s_len])
        attention_prob_t = nn.Softmax(dim=-1)(attention_score[:, :, :, s_len:])
        attention_prob = torch.cat((attention_prob_s, attention_prob_t), dim=-1)
        m_s = self.w_source_param * torch.ones(
            1, 1, s_len + t_len, s_len, device=attention_score.device
        )
        m_t = self.w_target_param * torch.ones(
            1, 1, s_len + t_len, t_len, device=attention_score.device
        )
        modality_mask = torch.cat((m_s, m_t), dim=3)
        attention_prob = self.dropout_1(attention_prob.mul(modality_mask))
        context_layer = (
            torch.matmul(attention_prob, v_layer).permute(0, 2, 1, 3).contiguous()
        )
        context_layer = self.dense(
            context_layer.view(context_layer.size(0), context_layer.size(1), -1)
        )
        hidden_state = self.LayerNorm1(self.dropout_2(context_layer) + fused_hidden)
        hidden_state = self.LayerNorm2(self.ff(hidden_state) + hidden_state)
        return hidden_state[:, :s_len, :], hidden_state[:, s_len:, :]


class MultimodalSelfAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _MultimodalCrossLayer(
                dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, source_x, target_x):
        for layer in self.layers:
            source_x, target_x = layer(source_x, target_x)
        return target_x


import torch.nn.functional as F


class ProjectorAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim,
        token_len,
        pooling_mode: str = "attention",
        use_expansion_mlp: bool = True,
    ):
        super().__init__()
        self.token_len = token_len
        self.token_dim = token_dim
        self.pooling_mode = pooling_mode
        self.use_expansion_mlp = use_expansion_mlp
        self.projection_head = nn.Linear(input_dim, token_dim)
        if self.pooling_mode == "attention":
            self.attention_net = nn.Sequential(
                nn.Linear(token_dim, token_dim // 2),
                nn.Tanh(),
                nn.Linear(token_dim // 2, 1),
            )
        if self.use_expansion_mlp:
            self.expansion_mlp = nn.Linear(token_dim, token_len * token_dim)

    def forward(self, embeddings, attention_mask):
        projected_embeddings = self.projection_head(embeddings)
        if self.pooling_mode == "attention":
            attention_scores = self.attention_net(projected_embeddings)
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(2) == 0, -1e9
            )
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(projected_embeddings * attention_weights, dim=1)
        elif self.pooling_mode == "cls":
            context_vector = projected_embeddings[:, 0, :]
        elif self.pooling_mode == "mean":
            masked_embeddings = projected_embeddings * attention_mask.unsqueeze(2)
            summed = torch.sum(masked_embeddings, dim=1)
            actual_lengths = attention_mask.sum(dim=1).unsqueeze(1)
            actual_lengths = torch.clamp(actual_lengths, min=1e-9)
            context_vector = summed / actual_lengths
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        if self.use_expansion_mlp:
            expanded_output = self.expansion_mlp(context_vector)
            output_seq = expanded_output.view(-1, self.token_len, self.token_dim)
        else:
            output_seq = context_vector.unsqueeze(1).repeat(1, self.token_len, 1)
        return output_seq


class CLFEFused(nn.Module):
    def __init__(self, args):
        super().__init__()
        config_model = args.model
        self.token_len = config_model.token_len
        fae_pooling_mode = getattr(config_model, "fae_pooling_mode", "attention")
        fae_use_expansion = getattr(config_model, "fae_use_expansion", True)
        use_channel_attention = getattr(config_model, "use_channel_attention", True)
        use_bert_finetune = getattr(config_model, "use_bert_finetune", True)
        self.use_chan_attn = use_channel_attention

        self.bertmodel = BertTextEncoder(
            use_finetune=use_bert_finetune, pretrained=config_model.bert_pretrained
        )
        self.proj_l = ProjectorAttention(
            input_dim=config_model.l_input_dim,
            token_dim=config_model.token_dim,
            token_len=config_model.token_len,
            pooling_mode=fae_pooling_mode,
            use_expansion_mlp=fae_use_expansion,
        )
        self.proj_a = nn.Sequential(
            nn.Linear(config_model.a_input_dim, config_model.a_proj_dst_dim),
            Transformer(
                num_frames=config_model.a_input_length,
                save_hidden=False,
                token_len=config_model.token_length,
                dim=config_model.proj_input_dim,
                depth=config_model.proj_depth,
                heads=config_model.proj_heads,
                mlp_dim=config_model.proj_mlp_dim,
            ),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(config_model.v_input_dim, config_model.v_proj_dst_dim),
            Transformer(
                num_frames=config_model.v_input_length,
                save_hidden=False,
                token_len=config_model.token_length,
                dim=config_model.proj_input_dim,
                depth=config_model.proj_depth,
                heads=config_model.proj_heads,
                mlp_dim=config_model.proj_mlp_dim,
            ),
        )
        if self.use_chan_attn:
            self.chan_attn_l = ChannelAttention(config_model.token_dim, ratio=8)
            self.chan_attn_a = ChannelAttention(config_model.token_dim, ratio=8)
            self.chan_attn_v = ChannelAttention(config_model.token_dim, ratio=8)

        self.l_encoder = Transformer(
            num_frames=config_model.token_length,
            save_hidden=True,
            token_len=None,
            dim=config_model.proj_input_dim,
            depth=config_model.AHL_depth,
            heads=config_model.l_enc_heads,
            mlp_dim=config_model.l_enc_mlp_dim,
        )
        self.guidance_encoder = HierarchicalGuidanceEncoder(
            dim=config_model.token_dim,
            depth=config_model.AHL_depth,
            heads=config_model.ahl_heads,
            dim_head=config_model.ahl_dim_head,
            dropout=config_model.ahl_droup,
        )
        self.final_fusion_layer = MultimodalSelfAttention(
            dim=config_model.proj_input_dim,
            depth=config_model.fusion_layer_depth,
            heads=config_model.fusion_heads,
            dim_head=config_model.ahl_dim_head,
            mlp_dim=config_model.fusion_mlp_dim,
            dropout=config_model.ahl_droup,
        )
        self.regression_layer = nn.Sequential(nn.Linear(config_model.token_dim, 1))

    def forward(self, x_visual, x_audio, x_text, vision_noise=None):
        text_attention_mask = x_text[:, 1, :]
        last_hidden_state, _ = self.bertmodel(x_text)
        if vision_noise is not None:
            x_visual = x_visual + vision_noise
        h_v_proj = self.proj_v(x_visual)[:, : self.token_len]
        h_a_proj = self.proj_a(x_audio)[:, : self.token_len]
        h_l_proj = self.proj_l(last_hidden_state, text_attention_mask)
        if self.use_chan_attn:
            h_l_initial = self.chan_attn_l(h_l_proj)
            h_a_initial = self.chan_attn_a(h_a_proj)
            h_v_initial = self.chan_attn_v(h_v_proj)
        else:
            h_l_initial = h_l_proj
            h_a_initial = h_a_proj
            h_v_initial = h_v_proj
        h_t_list = self.l_encoder(h_l_initial)
        h_hyper_guided = self.guidance_encoder(h_t_list, h_a_initial, h_v_initial)
        fused_output = self.final_fusion_layer(
            source_x=h_hyper_guided, target_x=h_t_list[-1]
        )
        feat = fused_output[:, 0]
        output = self.regression_layer(feat)
        return output


def build_model(args) -> nn.Module:
    return CLFEFused(args)
