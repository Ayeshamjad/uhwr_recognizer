import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool=None, drop=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(pool, pool))
        if drop > 0:
            layers.append(nn.Dropout2d(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class AuxEncoder(nn.Module):
    """
    Auxiliary encoder aligned with main CNNEncoder.
    Time axis = HEIGHT (same as main encoder)
    """
    def __init__(self, out_dim=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2, 2),          # H: 1600 → 800, W: 64 → 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),          # H: 800 → 400, W: 32 → 16
        )

        # 👇 NO HEIGHT POOLING AFTER THIS POINT
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),        # W: 16 → 8
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(48, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),        # W: 8 → 4
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 4),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),                   # W: 4 → 1
        )

    def forward(self, x):
        x = self.conv1(x)
        # print("Aux Encoder Conv1 output shape:", x.shape)
        x = self.conv2(x)
        # print("Aux Encoder Conv2 output shape:", x.shape)
        x = self.conv3(x)
        # print("Aux Encoder Conv3 output shape:", x.shape)
        x = self.conv4(x)
        # print("Aux Encoder Conv4 output shape:", x.shape)
        x = self.conv5(x)
        # print("Aux Encoder Conv5 output shape:", x.shape)

        # x: [B, C, H′, 1]
        x = x.squeeze(-1)               # [B, C, H′]
        # print("Aux Encoder after squeeze shape:", x.shape)
        x = x.permute(0, 2, 1)          # [B, H′, C]
        # print("Aux Encoder after permute shape:", x.shape)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_dim=256 + 4*64, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        

    def forward(self, feats):
        """
        feats: list of tensors [B, T, C_i]
        """
        # print("FeatureFusion input feature shapes:", [f.shape for f in feats])
        x = torch.cat(feats, dim=-1)      # [B, T, 512]
        # print("FeatureFusion after concat shape:", x.shape)
        x = self.proj(x)                  # [B, T, 256]
        # print("FeatureFusion after projection shape:", x.shape)
        return x
    
class MultiModalCNNEncoder(nn.Module):
    def __init__(self, cnn_encoder_path=None):
        super().__init__()
        self.img_enc = CNNEncoder()
        if cnn_encoder_path is not None:
            self.img_enc.load_state_dict(torch.load(cnn_encoder_path))
            print(f"Loaded CNN encoder weights from {cnn_encoder_path}")
        
        # self.thk_enc = AuxEncoder()
        self.press_enc = AuxEncoder()
        self.xt_enc  = AuxEncoder()
        self.yt_enc  = AuxEncoder()
        self.h_enc   = AuxEncoder()

        self.fusion = FeatureFusion()

    def forward(self, pixel_values, ablation_mode="all"):
        img = pixel_values['img']
        press = pixel_values['img_pressure']
        xt = pixel_values['img_x_tilt']
        yt = pixel_values['img_y_tilt']
        h = pixel_values['img_height']
        
        f_img = self.img_enc(img)
        f_press = self.press_enc(press)
        f_xt = self.xt_enc(xt)
        f_yt = self.yt_enc(yt)
        f_h = self.h_enc(h)

        zero_like = {
            "img": torch.zeros_like(f_img),
            "press": torch.zeros_like(f_press),
            "xt": torch.zeros_like(f_xt),
            "yt": torch.zeros_like(f_yt),
            "h": torch.zeros_like(f_h),
        }

        ablation_sets = {
            "all": [f_img, f_press, f_xt, f_yt, f_h],
            "no_press": [f_img, zero_like["press"], f_xt, f_yt, f_h],
            "no_xt": [f_img, f_press, zero_like["xt"], f_yt, f_h],
            "no_yt": [f_img, f_press, f_xt, zero_like["yt"], f_h],
            "no_h": [f_img, f_press, f_xt, f_yt, zero_like["h"]],
            "img_only": [f_img, zero_like["press"], zero_like["xt"], zero_like["yt"], zero_like["h"]],
        }

        fused_outputs = {name: self.fusion(parts) for name, parts in ablation_sets.items()}
        return fused_outputs[ablation_mode] if ablation_mode is not None else fused_outputs["all"]