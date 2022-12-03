import torch
import torch.nn as nn
import torch.nn.functional as F
        
class TimeEmbedding(nn.Module):
    def __init__(self, time_step, dim):
        super().__init__()
        pos = torch.arange(time_step)
        log_emb = torch.arange(0, dim, step=2) / dim* torch.log(torch.tensor(10000.0))
        emb = torch.exp(log_emb[None, :]) * pos[:, None]
        emb = torch.concat([emb.sin(), emb.cos()], dim=-1)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, x):
        return self.timembedding(x)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *_args):
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, *_args):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x, *_args):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim, dropout = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 4, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, pos):
        h = self.block1(x)
        h += self.temb_proj(pos)[..., None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)

        return h


class UNet(nn.Module):
    def __init__(self, input_shape, time_step, ch, num_res_blocks, ch_mults=(1, 2, 4), attn_res=(16, 8, 4)):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_step=time_step, dim=ch)

        self.head = nn.Conv2d(input_shape[0], ch, kernel_size=3, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        now_res = input_shape[1]

        for i, mult in enumerate(ch_mults):
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, dim=ch, dropout=0.1))
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mults) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
                now_res //= 2

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, dim=ch, dropout=0.1),
            AttnBlock(in_ch=now_ch),
            ResBlock(now_ch, now_ch, dim=ch, dropout=0.1),
        ])

        self.upblocks = nn.ModuleList()

        for i, mult in enumerate(reversed(ch_mults)):
            out_ch = ch * mult
            
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, dim=ch, dropout=0.1))
                now_ch = out_ch

                if now_res in attn_res:
                    self.upblocks.append(AttnBlock(now_ch))

            if i != len(ch_mults) - 1:
                self.upblocks.append(UpSample(in_ch=now_ch))
                now_res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, input_shape[0], kernel_size=1)
        )

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        x = self.head(x)

        xs = [x]
        for block in self.downblocks:
            x = block(x, temb)
            if not isinstance(block, AttnBlock):
                xs.append(x)

        # Middle
        for block in self.middleblocks:
            x = block(x, temb)

        # Upsampling
        for block in self.upblocks:
            if isinstance(block, ResBlock):
                x = torch.cat([x, xs.pop()], dim=1)
            x = block(x, temb)

        return self.out(x)