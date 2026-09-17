import torch
from transformers import RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
import torch.nn as nn
from torch import Tensor
from torch import nn
import math


def model_conv_transformer(inchannel, vocab_size, pretrained_decoder_path="./decoder_pretrain_proper/checkpoint-32452"):

    class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()            # 512 * 64

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.MaxPool2d(2, 2)  # 256 * 32
                # nn.MaxPool2d((2, 4), (2, 4)),   # 128 * 8
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2, 2)    # 128 * 16
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d((1, 2), (1, 2)),   # 128 * 8
                nn.Dropout2d(0.2),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d((1, 2), (1, 2)),    # 128 * 4
                nn.Dropout2d(0.2),
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, 4),
                nn.BatchNorm2d(256),
                nn.GELU(),
            )

        def forward(self,
                    src: Tensor,
                    ):

            src = self.conv1(src)
            # print(x.shape)                                 # (*, 16, 32, 256)
            src = self.conv2(src)
            # print(x.shape)                                 # (*, 32, 16, 128)
            src = self.conv3(src)
            # print(x.shape)                                 # (*, 64, 8, 128)
            src = self.conv4(src)
            # print(x.shape)                                 # (*, 128, 4, 128)
            src = self.conv5(src)
            # print(x.shape)                                 # (*, 256, 1, 125)
            src = src.squeeze(-1)
            src = src.permute((0, 2, 1)).contiguous()        # (*, 125, 256)

            return src

    model_conv = Conv()
    # model_conv = Easter2(inchannel, 256)

    dec = {'vocab_size': vocab_size,
           'n_positions': 512,
           'n_embd': 256,
           'n_head': 4,
           'n_layer': 2
           }

    dec = {'vocab_size': vocab_size,
           'n_positions': 512,
           'n_embd': 256,
           'n_head': 8,
           'n_layer': 3
           }

    enc = {'vocab_size': vocab_size,
           'num_hidden_layers': 2,
           'hidden_size': 256,
           'num_attention_heads': 4,
           'intermediate_size': 1024,
           'hidden_act': 'gelu'
           }

    enc = {'vocab_size': vocab_size,
           'num_hidden_layers': 3,
           'hidden_size': 256,
           'num_attention_heads': 8,
           'intermediate_size': 1024,
           'hidden_act': 'gelu'
           }

    enc_config = RobertaConfig(**enc)

    # dec_config = RobertaConfig(**enc)
    dec_config = GPT2Config(**dec)
    

    # dec_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    # enc_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        enc_config, dec_config)
    model_transformer = EncoderDecoderModel(config=config)
    
    model_transformer.decoder.load_state_dict(
    GPT2LMHeadModel.from_pretrained(pretrained_decoder_path).transformer.state_dict()
)

    return model_conv, model_transformer


def sinusoidal_positions(n_positions, dim):
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def decoder_only_model(vocab_size):

    dec_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=256,
        n_head=8,
        n_layer=3
    )

    decoder_lm = GPT2LMHeadModel(dec_config)

    # Replace decoder positional embedding
    decoder_lm.transformer.wpe = nn.Embedding.from_pretrained(
        sinusoidal_positions(dec_config.n_positions, dec_config.n_embd),
        freeze=True
    )

    return decoder_lm
