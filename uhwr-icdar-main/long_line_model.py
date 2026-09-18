"""UHWR-compatible model with extended visual positions and source masks.

Uses the learned GPT-2 positions employed by train_mixed.py, not its patched
sinusoidal alternative. CTC parameters are retained for checkpoint compatibility;
this first baseline trains the attention decoder with cross-entropy only.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from model.cnn_encoder import CNNEncoder
from model.ctc_head import CTCHead
from model.transformer_encoder import RobertaEncoder


class ModelConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


class LongLineModel(nn.Module):
    def __init__(self, vocab_size, bos, eos, pad, visual_positions=2048):
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, bos=bos, eos=eos, pad=pad,
                                  visual_positions=visual_positions, architecture='uhwr-mixed-long-line-v1')
        self.cnn_encoder = CNNEncoder()
        self.projection = nn.Identity()
        self.transformer_encoder = RobertaEncoder(256, 8, 3, 1024)
        encoder = self.transformer_encoder.transformer_encoder
        # inputs_embeds creates sequential RoBERTa IDs starting after padding_idx.
        offset = encoder.embeddings.position_embeddings.padding_idx + 1
        encoder.embeddings.position_embeddings = nn.Embedding(visual_positions + offset, 256,
                                                              padding_idx=offset-1)
        nn.init.normal_(encoder.embeddings.position_embeddings.weight, std=encoder.config.initializer_range)
        with torch.no_grad():
            encoder.embeddings.position_embeddings.weight[offset-1].zero_()
        encoder.config.max_position_embeddings = visual_positions + offset
        encoder.embeddings.register_buffer('position_ids', torch.arange(visual_positions+offset)[None], persistent=False)
        encoder.embeddings.register_buffer('token_type_ids', torch.zeros(1, visual_positions+offset, dtype=torch.long), persistent=False)
        self.transformer_encoder.pos_embedding.pos = nn.Parameter(torch.empty(1, visual_positions, 256))
        nn.init.trunc_normal_(self.transformer_encoder.pos_embedding.pos, std=0.02)
        self.transformer_decoder = GPT2LMHeadModel(GPT2Config(
            vocab_size=vocab_size, n_positions=512, n_embd=256, n_head=8, n_layer=3,
            bos_token_id=bos, eos_token_id=eos, pad_token_id=pad, add_cross_attention=True,
        ))
        self.ctc_head = CTCHead(256, vocab_size)
        self.ctc_head.requires_grad_(False)

    def initialize(self, checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'model' in state:
            state = state['model']
        own = self.state_dict()
        expandable = {
            'transformer_encoder.pos_embedding.pos': 1,
            'transformer_encoder.transformer_encoder.embeddings.position_embeddings.weight': 0,
        }
        for key, axis in expandable.items():
            if key not in state:
                raise ValueError(f'Checkpoint lacks {key}')
            old, new = state[key], own[key].clone()
            if old.ndim != new.ndim or any(old.size(i) != new.size(i) for i in range(old.ndim) if i != axis):
                raise ValueError(f'Incompatible checkpoint positional shape: {key}')
            if old.size(axis) > new.size(axis):
                raise ValueError(f'Checkpoint positional table is too large: {key}')
            selection = [slice(None)] * new.ndim
            selection[axis] = slice(0, old.size(axis))
            new[tuple(selection)] = old
            state[key] = new
        # Strict loading catches architecture/vocabulary mismatches instead of silently skipping.
        self.load_state_dict(state, strict=True)

    def encode(self, images, widths):
        features = self.cnn_encoder(images)
        lengths = (widths // 4 - 3).clamp(min=1, max=features.size(1))
        mask = torch.arange(features.size(1), device=images.device)[None] < lengths[:, None]
        return self.transformer_encoder(features, attention_mask=mask), mask

    def forward(self, images, labels, widths):
        memory, source_mask = self.encode(images, widths)
        return self.transformer_decoder(
            input_ids=labels, attention_mask=labels.ne(self.config.pad),
            encoder_hidden_states=memory, encoder_attention_mask=source_mask,
            use_cache=False,
        ).logits

    @torch.no_grad()
    def recognize(self, images, widths):
        memory, source_mask = self.encode(images, widths)
        prompt = torch.full((images.size(0), 1), self.config.bos, device=images.device, dtype=torch.long)
        return self.transformer_decoder.generate(
            input_ids=prompt, attention_mask=torch.ones_like(prompt),
            encoder_hidden_states=memory, encoder_attention_mask=source_mask,
            max_new_tokens=511, do_sample=False, num_beams=1, use_cache=True,
            bos_token_id=self.config.bos, eos_token_id=self.config.eos, pad_token_id=self.config.pad,
            suppress_tokens=[self.config.bos, self.config.pad],
        )[:, 1:]

    def save_pretrained(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        (path/'config.json').write_text(json.dumps(self.config.to_dict(), indent=2))
        torch.save(self.state_dict(), path/'model.pt')

    @classmethod
    def from_pretrained(cls, path):
        config = json.loads((Path(path)/'config.json').read_text())
        config.pop('architecture')
        model = cls(**config)
        model.load_state_dict(torch.load(Path(path)/'model.pt', map_location='cpu', weights_only=True))
        return model
