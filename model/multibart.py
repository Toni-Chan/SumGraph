"""PyTorch BART model, ported from the fairseq repo."""
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

# Transformer definitions

from transformers.activations import ACT2FN
from transformers.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.utils import logging
from transformers import BartForConditionalGeneration, BartTokenizer 
from transformers.modeling_bart import PretrainedBartModel
# from generation_utils import BartGenerationMixin  # generation_utils changed based on multi-bart

# local functions


from model.graph_enc import graph_encode, gat_encode, node_mask, subgraph_encode

from .util import (
    _make_linear_from_emb,
    invert_mask,
    LayerNorm,
    _prepare_bart_decoder_inputs,
    len_mask,
    sequence_mean,
    pad_batch_tensorize
)
from model.extract import MeanSentEncoder
MAX_FREQ = 100

MULTI_INPUTS=['source','graph']

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
]

######## Helper modules ########
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        
            
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
            # saved_state = None
            
        else:
            # this branch is hit by encoder
            saved_state = None
        
        q = self.q_proj(query) * self.scaling
        if static_kv and key is None:  # cross-attention with cache
            k = v = None
        elif static_kv and key is not None:  # cross-attention no prev_key found in cache
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:  # self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state:
            k, v = self._concat_saved_state(k, v, saved_state, static_kv, bsz)

        # Update cache
        if isinstance(layer_state, dict):
            cached_shape = (bsz, self.num_heads, -1, self.head_dim)  # bsz must be first for reorder_cache
            layer_state[self.cache_key] = dict(prev_key=k.view(*cached_shape), prev_value=v.view(*cached_shape))

        src_len = k.size(1)
        assert key_padding_mask is None or key_padding_mask.shape == (bsz, src_len)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Note: deleted workaround to get around fork/join parallelism not supporting Optional types. on 2020/10/15

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _concat_saved_state(self, k, v, saved_state, static_kv, bsz) -> Tuple[Tensor]:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        prev_K = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
        prev_V = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
        new_K = prev_K if static_kv else torch.cat([prev_K, k], dim=1)
        new_V = prev_V if static_kv else torch.cat([prev_V, v], dim=1)
        return new_K, new_V


class multiSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: dict() = None
    encoder_hidden_states: dict() = None
    encoder_attentions: dict() = None


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

######## Encoder-Decoder ########

class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens

        Returns:
            BaseModelOutput or Tuple comprised of:

                - **x** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`
                - **encoder_states** (tuple(torch.FloatTensor)): all intermediate hidden states of shape `(src_len,
                  batch, embed_dim)`. Only populated if *output_hidden_states:* is True.
                - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class multiDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        
        
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_persona =  Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_intent =  Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_intent.cache_key = "encoder_decoder_i" 
        
        self.encoder_attn_persona.cache_key = "encoder_decoder_p" 
        
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        
        self.weight_attn = nn.Linear(3*self.embed_dim, self.embed_dim)
        # self.weight_attn = torch.nn.Parameter(torch.ones(len(MULTI_INPUTS), 1) / len(MULTI_INPUTS))
        
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_dict = {'persona':self.encoder_attn_persona,'source':self.encoder_attn,'event':self.encoder_attn_intent}
        

        
    def reload_module_dict(self):
        self.encoder_attn_intent = deepcopy(self.encoder_attn)
        self.encoder_attn_intent.cache_key = "encoder_decoder_i" 
        self.encoder_attn_persona = deepcopy(self.encoder_attn)
        self.encoder_attn_persona.cache_key = "encoder_decoder_p" 
        self.encoder_dict = {'persona':self.encoder_attn_persona,'source':self.encoder_attn,'event':self.encoder_attn_intent}
        
            
    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross-Attention Block
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        cross_attn_weights = {}
        x_attn_output ={}
        for key in MULTI_INPUTS:
            x_attn_output[key],cross_attn_weights[key] = self.encoder_dict[key](
                query=x,
                key=encoder_hidden_states[key],
                key_padding_mask=encoder_attn_mask[key],
                layer_state=layer_state,  # mutates layer state
                output_attentions=output_attentions,
            )
        # x = torch.mean(torch.stack([x_attn_output[key] for key in MULTI_INPUTS]) * self.weight_attn.unsqueeze(-1).unsqueeze(-1), dim=0)
        x = self.weight_attn(torch.cat([x_attn_output[key] for key in MULTI_INPUTS], dim=-1))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
            cross_attn_weights,
        )  # layer_state = cache for decoding


class multiBartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`DecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.do_blenderbot_90_layernorm = config.do_blenderbot_90_layernorm  # layernorm variant
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [multiDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None
    def reload_module_dict(self):
        for layer in self.layers:
            layer.reload_module_dict()

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,
        EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation

        Returns:
            BaseModelOutputWithPast or tuple:

                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")

        # check attention mask and invert
        encoder_padding_mask_inverted = {}
        if encoder_padding_mask is not None:
            for k in encoder_padding_mask.keys():
                encoder_padding_mask_inverted[k] = invert_mask(encoder_padding_mask[k])

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.do_blenderbot_90_layernorm:
            x = self.layernorm_embedding(x)
            x += positions
        else:
            x += positions
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        for key in MULTI_INPUTS:
            encoder_hidden_states[key] = encoder_hidden_states[key].transpose(0, 1)
            
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache: List[Dict] = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = past_key_values[idx] if past_key_values is not None else None

            x, layer_self_attn, layer_past, layer_cross_attn = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask_inverted,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attentions += (layer_cross_attn,)

        if self.layer_norm:  # if config.add_final_layer_norm (mBART)
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        for key in MULTI_INPUTS:
            encoder_hidden_states[key] = encoder_hidden_states[key].transpose(0, 1)
          

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in [x, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class multiSeq2SeqModelOutput(Seq2SeqModelOutput):

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: dict()
    encoder_hidden_states: dict()
    encoder_attentions: dict()

######## Usage of BART: encoder, decoder ########
class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class multiBartGAT(PretrainedBartModel):
    def __init__(self, config: BartConfig, gat_args):
        """Initialize the GAT-multi-BART model.

        Args:
            config (BartConfig): BART parameters
            gat_args (Dict): GAT parameters
        """
        super().__init__(config)

        # BART settings
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # GAT settings
        feat_emb_dim = config.d_model // 4
        graph_hsz = 0
        self.node_freq = gat_args['node_freq']
        if gat_args['node_freq']:
            graph_hsz += feat_emb_dim
            self._node_freq_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        gat_args['graph_hsz'] = graph_hsz

        self.graph_enc = subgraph_encode(gat_args)
        self.node_enc = MeanSentEncoder()

        mask_type = gat_args['mask_type']
        
        if mask_type == 'encoder':
            self._graph_mask = node_mask(mask_type='gold')
        elif mask_type == 'soft':
            self._graph_mask = node_mask(mask_type=mask_type, emb_dim=graph_hsz)
        else:
            self._graph_mask = node_mask(mask_type='none')

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.init_weights()



    def forward(self, article, art_lens, abstract, extend_art, extend_vsize, ninfo, rinfo, ext_ninfo=None):
        """Forward pass of the whole encoder-decoder process.

        Args:
            article (Tensor): article tensor token ids
            art_lens (Tensor): article lengths
            abstract (Tensor): [description]
            extend_art ([type]): [description]
            extend_vsize ([type]): [description]
            ninfo (tuple): (nodes, nmask, node_num, sw_mask, feature_dict, node_lists) for nodes information of the graph
            rinfo (tuple): (relations, rmask, triples, adjs) for relations information of the graph
            ext_ninfo ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        (nodes, nmask, node_num, sw_mask, feature_dict, node_lists) = ninfo
        (relations, rmask, triples, adjs) = rinfo

        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids=abstract,
                causal_mask_dtype=self.shared.weight.dtype,
            )

        assert decoder_input_ids is not None

        ### Forward pass
        # 1. Load text to encoder 
        last_hidden,hidden_states,attention = self.encoder(article)
        # 2. Use encoded text and graph input to from attention
        
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None

        if self._mask_type == 'soft' or self._mask_type == 'none':
            outputs = self._encode_graph(hidden_states, nodes, nmask, relations,
                                              rmask, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        else:
            outputs = self._encode_graph(hidden_states, nodes, nmask, relations, rmask, adjs, node_lists, sw_mask,
                                       nodefreq=feature_dict['node_freq'])
        
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None
        ext_info = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # send the source to BART again?

        # graph_encoded = self.encoder(topics, attention_mask=masks)
        # new_last_hidden = graph_encoded.last_hidden_state
        # new_attention = graph_encoded.attentions
        # new_hidden_states = graph_encoded.hidden_states
        
        # 3. Load to decoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            topics,
            encoder_padding_mask=None,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            use_cache=False
        )

        final_state = decoder_outputs[0]

        lm_logits = F.linear(final_state, self.shared.weight, bias=self.final_logits_bias)
        return lm_logits

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, batch_adjs, node_lists, node_mask=None, nodefreq=None):
        d_word = articles.size(-1)

        masks = []
        bs, n_node, n_word = nodes.size()
        nodes = nodes.view(bs, -1).unsqueeze(2).expand(bs, n_node * n_word, d_word)
        nodes = articles.gather(1, nodes).view(bs, n_node, n_word, d_word).contiguous()
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)
        nodes = self.node_enc(nodes, mask=nmask)
        if self.node_freq:
            assert nodefreq is not None
            nodefreq = self._node_freq_embedding(nodefreq)
            nodes = torch.cat([nodes, nodefreq], dim=-1)
        if self._mask_type == 'encoder':
            nodes, node_mask = self._graph_mask(nodes, node_mask)
        elif self._mask_type == 'soft':
            nodes, node_mask = self._graph_mask(nodes, _input=nodes)
            masks.append(node_mask.squeeze(2))

        # topics, topic_length = self._graph_enc(batch_adjs, nodes, node_lists)
        topics, topic_length = self.graph_enc(batch_adjs, nodes, node_lists)

        results = ((topics, topic_length),)

        if 'soft' in self._mask_type:
            results += (masks,)

        return results
        # if 'soft' in self._mask_type:
        #     return (topics, topic_length), masks
        # else:
        #     return (topics, topic_length)


    def greedy(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, node_lists, adjs,
                     go, eos, unk, max_len, tar_in):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings

        # 1. send to encode
        last_hidden,hidden_states,attention = self.encoder(article)

        # 2. GAT
        outputs = self._encode_graph(hidden_states, nodes, nmask, None,
                                              None, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # graph_encoded = self.encoder(topics, attention_mask=masks)
        # new_last_hidden = graph_encoded.last_hidden_state
        # new_attention = graph_encoded.attentions
        # new_hidden_states = graph_encoded.hidden_states

        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        for i in range(max_len):
            decoder_outputs = self.decoder(
                tok,
                topics,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                use_cache=False
                )
            
            new_last_hidden = decoder_outputs[0]
            logits = F.linear(new_last_hidden, self.shared.weight, bias=self.final_logits_bias)
            attn_score = decoder_outputs.attention
            
            # Select out tokens
            tok = torch.max(logits, dim=1, keepdim=True)[1]
            
            #print('greedy tok:', tok)
            if i == 0:
                unfinished = (tok != eos)
                #print('greedy tok:', tok)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score)
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns

    def sample(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, node_lists, adjs,
                     go, eos, unk, max_len, abstract, ml):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings

        # 1. send to encode
        last_hidden,hidden_states,attention = self.encoder(article)

        # 2. GAT
        outputs = self._encode_graph(hidden_states, nodes, nmask, None,
                                              None, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # graph_encoded = self.encoder(topics, attention_mask=masks)
        # new_last_hidden = graph_encoded.last_hidden_state
        # new_attention = graph_encoded.attentions
        # new_hidden_states = graph_encoded.hidden_states

        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        seqLogProbs = []
        for i in range(max_len):
            
            decoder_outputs = self.decoder(
                tok,
                topics,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                use_cache=False
                )
            
            new_last_hidden = decoder_outputs[0]
            logits = F.linear(new_last_hidden, self.shared.weight, bias=self.final_logits_bias)
            attn_score = decoder_outputs[3]
            
            # also save log probability
            logprob = F.log_softmax(logits, dim=1)
            #print('logit:', logit.size())
            #score = F.softmax(logit, dim=1)
            score = torch.exp(logprob)
            tok = torch.multinomial(score, 1).detach()
            #print('out:', tok)
            sampleProb = logprob.gather(1, tok)
            seqLogProbs.append(sampleProb)
            
            #print('greedy tok:', tok.size())
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score)
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns, seqLogProbs

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     ninfo, rinfo, ext_ninfo,
                     go, eos, unk, max_len, beam_size, diverse=1.0, min_len=0):
        """ greedy decode support batching"""

        return 0

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings

        # 1. send to encode
        last_hidden,hidden_states,attention = self.encoder(article)

        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        for i in range(max_len):
            decoder_outputs = self.decoder(
                tok,
                last_hidden,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                use_cache=False
                )
            
            new_last_hidden = decoder_outputs[0]
            logits = F.linear(new_last_hidden, self.shared.weight, bias=self.final_logits_bias)
            attn_score = decoder_outputs[3]
            
            # Select out tokens
            tok = torch.max(logits, dim=1, keepdim=True)[1]
            
            #print('greedy tok:', tok)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk

        return outputs, attns
    
    # def batched_beamsearch(self, article, art_lens,
    #                        extend_art, extend_vsize,
    #                        ninfo, rinfo, ext_ninfo,
    #                        go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
    #     (nodes, nmask, node_num, sw_mask, feature_dict, node_lists) = ninfo
    #     (relations, rmask, triples, adjs) = rinfo
    #     if self._copy_from_node:
    #         (all_node_words, all_node_mask, ext_node_aligns, gold_copy_mask) = ext_ninfo
    #     if self._gold:
    #         sw_mask = sw_mask
    #     else:
    #         sw_mask = None
    #     batch_size = len(art_lens)
    #     vsize = self._embedding.num_embeddings
        
    #     last_hidden,hidden_states,attention  = self.encoder(article)

    #     if self._mask_type == 'soft' or self._mask_type == 'none':
    #         outputs = self._encode_graph(attention, nodes, nmask, relations,
    #                                           rmask, adjs, node_lists, node_mask=None,
    #                                           nodefreq=feature_dict['node_freq'])
    #     else:
    #         outputs = self._encode_graph(attention, nodes, nmask, relations, rmask, adjs, node_lists, sw_mask,
    #                                    nodefreq=feature_dict['node_freq'])
    #     if self._hierarchical_attn:
    #         topics, masks, paras = outputs
    #     elif 'soft' in self._mask_type:
    #         topics, masks = outputs
    #         nodes = topics[0]
    #         node_num = topics[1]
    #         paras = None
    #     else:
    #         topics = outputs
    #         nodes = topics[0]
    #         node_num = topics[1]
    #         paras = None
    #     ext_info = None

    #     mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        
    #     # graph_encoded = self.encoder(topics, attention_mask=masks)
    #     # new_last_hidden = graph_encoded.last_hidden_state
    #     # new_attention = graph_encoded.attentions
    #     # new_hidden_states = graph_encoded.hidden_states
        
    #     ## decoding
    #     # init values

    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     max_length = max_length if max_length is not None else self.config.max_length
    #     pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    #     eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    #     batch_size = len(beam_scorer._beam_hyps)
    #     num_beams = beam_scorer.num_beams

    #     batch_beam_size, cur_len = input_ids.shape
    #     assert (
    #         num_beams * batch_size == batch_beam_size
    #     ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    #     beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    #     beam_scores[:, 1:] = -1e9
    #     beam_scores = beam_scores.view((batch_size * num_beams,))

    #     while cur_len < max_length:
    #         decoder_outputs = self.decoder(
    #             tok,
    #             topics,
    #             encoder_padding_mask=None,
    #             decoder_padding_mask=None,
    #             decoder_causal_mask=None,
    #             use_cache=False
    #             )
            
    #         new_last_hidden = decoder_outputs.last_hidden_state
    #         logits = F.linear(new_last_hidden, self.shared.weight, bias=self.final_logits_bias)
    #         attn_score = decoder_outputs.attention
            
    #         # also save log probability
    #         logprob = F.log_softmax(logits, dim=1)
    #         next_token_logits = outputs.logits[:, -1, :]

    #         # adjust tokens for Bart, *e.g.*
    #         next_token_logits = self.adjust_logits_during_generation(
    #             next_token_logits, cur_len=cur_len, max_length=max_length
    #         )

    #         next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

    #         next_token_scores = logits_processor(input_ids, next_token_scores)
    #         next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    #         # reshape for beam search
    #         vocab_size = next_token_scores.shape[-1]
    #         next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    #         next_token_scores, next_tokens = torch.topk(
    #             next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    #         )

    #         next_indices = next_tokens // vocab_size
    #         next_tokens = next_tokens % vocab_size

    #         # stateless
    #         beam_outputs = beam_scorer.process(
    #             input_ids,
    #             next_token_scores,
    #             next_tokens,
    #             next_indices,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #         )
    #         beam_scores = beam_outputs["next_beam_scores"]
    #         beam_next_tokens = beam_outputs["next_beam_tokens"]
    #         beam_idx = beam_outputs["next_beam_indices"]

    #         input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    #         cur_len = cur_len + 1

    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         )
    #         if model_kwargs["past"] is not None:
    #             model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

    #         if beam_scorer.is_done:
    #             break

    #     decoded = beam_scorer.finalize(
    #         input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    #     )

    #     return decoded