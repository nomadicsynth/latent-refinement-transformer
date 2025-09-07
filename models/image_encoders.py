import torch, torch.nn as nn
import math

class ConvStemEncoder(nn.Module):
    def __init__(self, in_ch=1, hidden=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 14x14
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.ReLU() # 7x7
        )
        self.proj = nn.Linear(hidden, out_dim)
    def forward(self, x):  # x: (B,1,28,28)
        f = self.net(x)          # (B,C,7,7)
        f = f.permute(0,2,3,1).reshape(x.size(0), -1, f.size(1))  # (B,49,C)
        return self.proj(f)      # (B,49,out_dim)

class PatchEmbeddingEncoder(nn.Module):
    def __init__(self, img_size=28, patch=4, in_ch=1, out_dim=256):
        super().__init__()
        assert img_size % patch == 0
        self.conv = nn.Conv2d(in_ch, out_dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size // patch) ** 2
    def forward(self, x):
        f = self.conv(x)  # (B,out_dim,H',W')
        f = f.flatten(2).transpose(1,2)  # (B,N,out_dim)
        return f

class WaveletEncoder(nn.Module):
    def __init__(self, wave='haar', levels=2, out_dim=256):
        super().__init__()
        try:
            import pywt  # noqa
        except ImportError:
            raise RuntimeError("Install pywt for WaveletEncoder: pip install PyWavelets")
        self.wave = wave
        self.levels = levels
        self.out_dim = out_dim
        # simple per-coefficient linear projection after stacking
        self.proj = None  # lazy init
    def forward(self, x):
        # x: (B,1,H,W)
        import pywt
        B = x.size(0)
        coeff_tokens = []
        for b in range(B):
            arr = x[b,0].cpu().numpy()
            coeffs = pywt.wavedec2(arr, self.wave, level=self.levels)
            # coeffs: (LL_L, (LH_L, HL_L, HH_L), ..., (LH_1,HL_1,HH_1))
            flat_feats = []
            for i, c in enumerate(coeffs):
                if isinstance(c, tuple):
                    for sub in c:
                        flat_feats.append(torch.tensor(sub))
                else:
                    flat_feats.append(torch.tensor(c))
            # Resize each map to a common spatial size via interpolation then pool
            proc = []
            for m in flat_feats:
                t = m.unsqueeze(0).unsqueeze(0).float()  # (1,1,h,w)
                t = torch.nn.functional.interpolate(t, size=(8,8), mode='bilinear', align_corners=False)
                proc.append(t.squeeze(0))
            stacked = torch.cat(proc, dim=0)  # (C,8,8)
            coeff_tokens.append(stacked)
        feat = torch.stack(coeff_tokens, dim=0)  # (B,C,8,8)
        feat = feat.flatten(2).transpose(1,2)    # (B,N,C)
        if self.proj is None:
            self.proj = torch.nn.Linear(feat.size(-1), self.out_dim).to(feat.device)
        return self.proj(feat)

class SIRENImplicitEncoder(nn.Module):
    def __init__(self, img_size=28, hidden=128, layers=3, out_dim=256):
        super().__init__()
        self.img_size = img_size
        mods = []
        in_dim = 2
        for i in range(layers):
            lin = nn.Linear(in_dim if i==0 else hidden, hidden)
            nn.init.uniform_(lin.weight, -math.sqrt(6/in_dim)/30, math.sqrt(6/in_dim)/30)
            mods.append(lin)
            in_dim = hidden
        self.mlp = nn.ModuleList(mods)
        self.final = nn.Linear(hidden, out_dim)
    def forward(self, x):
        # x ignored except for batch size; regenerate coords each forward
        B = x.size(0)
        ys, xs = torch.meshgrid(
            torch.linspace(-1,1,self.img_size, device=x.device),
            torch.linspace(-1,1,self.img_size, device=x.device),
            indexing="ij"
        )
        coords = torch.stack([xs, ys], dim=-1).view(-1,2)  # (H*W,2)
        feats = coords
        for lin in self.mlp:
            feats = torch.sin(lin(feats))
        feats = self.final(feats)  # (H*W,out_dim)
        feats = feats.unsqueeze(0).expand(B, -1, -1)  # (B,N,out_dim)
        return feats