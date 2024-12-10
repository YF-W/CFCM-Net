import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as resnet_model

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes=None, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., new_shape=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = PatchEmbed(img_size=pair(image_size), patch_size=pair(patch_size), in_c=channels, embed_dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        if num_classes is not None:
            self.conv_head = nn.Conv2d(dim, num_classes, kernel_size=1)
        else:
            self.conv_head = nn.Identity()
        if new_shape is not None:
            self.upsample = nn.Upsample(size=(new_shape, new_shape), mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        new_dim = int(n ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_dim, w=new_dim)

        x = self.upsample(x)
        x = self.conv_head(x)

        return x

# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.double_conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VerticalImageSplitter(nn.Module):
    def __init__(self):
        super(VerticalImageSplitter, self).__init__()

    def forward(self, x_c, x_v):
        _, c1, h1, w1 = x_c.shape
        _, c2, h2, w2 = x_v.shape

        mask_left1 = torch.zeros(h1, w1).to(x_c.device)
        mask_left2 = torch.zeros(h2, w2).to(x_v.device)
        mask_right1 = torch.zeros(h1, w1).to(x_c.device)
        mask_right2 = torch.zeros(h2, w2).to(x_v.device)

        mask_left1[:, :w1//2] = 1
        mask_left2[:, :w2//2] = 1

        mask_right1[:, w1//2:] = 1
        mask_right2[:, w2//2:] = 1

        mask_left1 = mask_left1.unsqueeze(0).unsqueeze(0).expand(1, c1, h1, w1)
        mask_right1 = mask_right1.unsqueeze(0).unsqueeze(0).expand(1, c1, h1, w1)
        mask_left2 = mask_left2.unsqueeze(0).unsqueeze(0).expand(1, c2, h2, w2)
        mask_right2 = mask_right2.unsqueeze(0).unsqueeze(0).expand(1, c2, h2, w2)

        left_half1 = x_c * mask_left1
        right_half1 = x_c * mask_right1
        left_half2 = x_v * mask_left2
        right_half2 = x_v * mask_right2

        return left_half1, right_half1, left_half2, right_half2

class SymmetricImageSplitter(nn.Module):
    def __init__(self):
        super(SymmetricImageSplitter, self).__init__()

    def forward(self, x_c, x_v):
        _, c1, h1, w1 = x_c.shape
        _, c2, h2, w2 = x_v.shape

        mid_w1 = w1 // 2
        mid_w2 = w2 // 2

        left_half1 = x_c[:, :, :, :mid_w1]
        right_half2 = x_v[:, :, :, mid_w2:]

        flipped_left = torch.flip(left_half1, [3])

        flipped_right = torch.flip(right_half2, [3])

        x1 = torch.cat((left_half1, flipped_left), dim=3)

        x2 = torch.cat((flipped_right, right_half2), dim=3)

        return x1, x2


class CenterCropper(nn.Module):
    def __init__(self):
        super(CenterCropper, self).__init__()

    def forward(self, x_c, x_v):
        _, _, h1, w1 = x_c.shape
        _, _, h2, w2 = x_v.shape

        start_h1, start_w1 = h1 // 4, w1 // 4
        end_h1, end_w1 = start_h1 + h1 // 2, start_w1 + w1 // 2

        start_h2, start_w2 = h2 // 4, w2 // 4
        end_h2, end_w2 = start_h2 + h2 // 2, start_w2 + w2 // 2

        x1_center = x_c[:, :, start_h1:end_h1, start_w1:end_w1]
        x2_center = x_v[:, :, start_h2:end_h2, start_w2:end_w2]

        pad_left1, pad_right1 = start_w1, w1 - end_w1
        pad_top1, pad_bottom1 = start_h1, h1 - end_h1

        pad_left2, pad_right2 = start_w2, w2 - end_w2
        pad_top2, pad_bottom2 = start_h2, h2 - end_h2

        x1_padded = F.pad(x1_center, (pad_left1, pad_right1, pad_top1, pad_bottom1), mode='constant', value=0)
        x2_padded = F.pad(x2_center, (pad_left2, pad_right2, pad_top2, pad_bottom2), mode='constant', value=0)

        return x1_padded, x2_padded


class LeftRightSymmetryProcessor(nn.Module):
    def __init__(self):
        super(LeftRightSymmetryProcessor, self).__init__()

    def forward(self, x1, x2):
        _, c1, h1, w1 = x1.shape
        _, c2, h2, w2 = x2.shape

        mid_w1 = w1 // 2
        mid_w2 = w2 // 2

        right_half2 = x2[:, :, :, mid_w2:]
        flipped_right2 = torch.flip(right_half2, [3])
        x2_processed = torch.cat((flipped_right2, right_half2), dim=3)

        left_half1 = x1[:, :, :, :mid_w1]
        flipped_left1 = torch.flip(left_half1, [3])
        x1_processed = torch.cat((left_half1, flipped_left1), dim=3)

        return x1_processed, x2_processed

class CV_skips(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.dw_conv1 = DepthwiseSeparableConv(channel, channel)
        self.dw_conv2 = DepthwiseSeparableConv(channel, channel)
        self.dw_conv3 = DepthwiseSeparableConv(channel, channel)


        self.diagonal = VerticalImageSplitter()
        self.symmetric = SymmetricImageSplitter()
        self.center = CenterCropper()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.LeftRight = LeftRightSymmetryProcessor()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_c, x_v):
        x_c1, x_co, x_v1, x_vo = self.diagonal(x_c, x_v)
        x1, x2 =self.center(x_c1, x_v1)
        x = x1 + x2

        x_c2, x_v2 = self.symmetric(x_c1, x_v1)
        x_c3 = self.dw_conv1(x_c2)
        x_v3 = self.dw_conv2(x_v2)
        x_A = self.gap(x_c3)
        x_M = self.gmp(x_v3)
        x_A = self.sigmoid(x_A)
        x_M = self.sigmoid(x_M)
        x_c4 = x_A * x_c3
        x_v4 = x_M * x_v3

        x_c5, x_v5 = self.LeftRight(x_c4, x_v4)
        x_out = x_c5 + x_v5
        x_out = self.dw_conv3(x_out)
        x_out = x_out + x
        x_out = self.relu(x_out)
        x_c6 = x_c5 + x_co
        x_v6 = x_v5 + x_vo

        return x_c6, x_v6, x_out
# ==========================================
class CMT(nn.Module):
    def __init__(self, out_channels):
        super(CMT, self).__init__()
        self.conv_color_1 = nn.Sequential(
            nn.Conv2d(6, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )
        self.conv_color_2 = nn.Sequential(
            nn.Conv2d(6, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_light = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )
        self.conv_s = nn.Conv2d(1, out_channels // 2, 3, 1, 1)
        self.conv_sc = nn.Conv2d(out_channels, out_channels * 2 // 3, 3, 1, 1)
        self.conv_rgb = nn.Conv2d(3, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def rgb_to_lab(self, rgb):
        rgb = rgb.float() / 255.0
        # RGB to XYZ
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        r = torch.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
        g = torch.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
        b = torch.where(b <= 0.04045, b / 12.92, ((b + 0.055) / 1.055) ** 2.4)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        x = x / 0.95047
        y = y / 1.0
        z = z / 1.08883
        l = torch.where(y > 0.008856, (116 * (y ** (1 / 3))) - 16, 903.3 * y)
        a = 500 * (torch.where(x > 0.008856, (x ** (1 / 3)) - (y ** (1 / 3)), 0))
        b = 200 * (torch.where(z > 0.008856, (y ** (1 / 3)) - (z ** (1 / 3)), 0))
        return torch.cat([l.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)], dim=1)

    def rgb_to_hsv(self, rgb):
        rgb = rgb.float() / 255.0
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        cmax, _ = torch.max(rgb, dim=1)
        cmin, _ = torch.min(rgb, dim=1)
        delta = cmax - cmin
        h = torch.zeros_like(cmax)
        mask = delta != 0
        r_mask, g_mask, b_mask = (cmax == r) & mask, (cmax == g) & mask, (cmax == b) & mask
        h[r_mask] = (g[r_mask] - b[r_mask]) / delta[r_mask]
        h[g_mask] = (b[g_mask] - r[g_mask]) / delta[g_mask] + 2
        h[b_mask] = (r[b_mask] - g[b_mask]) / delta[b_mask] + 4
        h = (h / 6) % 1.0
        v = cmax
        s = torch.zeros_like(cmax)
        s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
        return torch.cat([h.unsqueeze(1), s.unsqueeze(1), v.unsqueeze(1)], dim=1)

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        h = h.float()
        s = s.float()
        v = v.float()

        hi = (h * 6).long() % 6
        f = (h * 6) - hi.float()
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        r = torch.zeros_like(h, dtype=torch.float32)
        g = torch.zeros_like(h, dtype=torch.float32)
        b = torch.zeros_like(h, dtype=torch.float32)

        r[hi == 0] = v[hi == 0]
        g[hi == 0] = t[hi == 0]
        b[hi == 0] = p[hi == 0]

        r[hi == 1] = q[hi == 1]
        g[hi == 1] = v[hi == 1]
        b[hi == 1] = p[hi == 1]

        r[hi == 2] = p[hi == 2]
        g[hi == 2] = v[hi == 2]
        b[hi == 2] = t[hi == 2]

        r[hi == 3] = p[hi == 3]
        g[hi == 3] = q[hi == 3]
        b[hi == 3] = v[hi == 3]

        r[hi == 4] = t[hi == 4]
        g[hi == 4] = p[hi == 4]
        b[hi == 4] = v[hi == 4]

        r[hi == 5] = v[hi == 5]
        g[hi == 5] = p[hi == 5]
        b[hi == 5] = q[hi == 5]

        rgb = torch.stack([r, g, b], dim=1)  # [N, 3, H, W]

        return rgb

    def forward(self, img):
        c = img.shape[1]
        if c != 3:
            conv_layer = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=1, stride=1).to('cuda:0')
            img_rgb = conv_layer(img)
        else:
            img_rgb = img

        img_rgb_np = (img_rgb * 255).clamp(0, 255).byte()

        img_lab = self.rgb_to_lab(img_rgb_np)
        img_hsv = self.rgb_to_hsv(img_rgb_np)

        img_l = img_lab[:, 0, :, :].unsqueeze(1)
        img_ab = img_lab[:, 1:, :, :]
        img_h = img_hsv[:, 0, :, :].unsqueeze(1)
        img_s = img_hsv[:, 1, :, :].unsqueeze(1)
        img_v = img_hsv[:, 2, :, :].unsqueeze(1)

        img_color = torch.cat((img_rgb, img_h, img_ab), dim=1)
        img_light = torch.cat((img_v, img_l), dim=1)

        img_s = img_s.float()
        img_light = img_light.float()
        img_color = img_color.float()
        img_color = img_color / 255.0

        img_color_1 = self.conv_color_1(img_color)
        img_color_2 = self.conv_color_2(img_color)
        img_light = self.conv_light(img_light)
        img_s = self.conv_s(img_s)

        img_sc = torch.cat((img_s, img_color_2), dim=1)
        img_sc = self.conv_sc(img_sc)
        c = img_sc.shape[1]

        if c % 2 != 0:
            c_even = c - 1
            first_half = img_sc[:, :c_even // 2, :, :]
            second_half = img_sc[:, c_even // 2:c_even, :, :]
        else:
            first_half = img_sc[:, :c // 2, :, :]
            second_half = img_sc[:, c // 2:, :, :]

        second_half = second_half + img_color_1
        img_cs_1 = torch.cat((second_half, first_half), dim=1)

        img_hsv_1 = torch.cat((img_cs_1, img_light), dim=1)
        c_hsv = img_hsv_1.shape[1]

        conv_img_hsv_2 = nn.Conv2d(c_hsv, 3, 3, 1, 1).to('cuda:0')
        img_hsv_2 = conv_img_hsv_2(img_hsv_1)
        img_hsv_3 = img_hsv_2 + img_hsv
        img_rgb2 = self.hsv_to_rgb(img_hsv_3)
        img_rgb_2 = (img_rgb2 * 255).clamp(0, 255).byte()
        img_rgb_2 = img_rgb_2.float() / 255.0
        img_rgb_3 = self.conv_rgb(img_rgb_2)

        out = img_rgb_3 + img

        return out
# =========================================
class CFCM_Net(nn.Module):
    def __init__(self):
        super(CFCM_Net, self).__init__()

        self.c_down = nn.MaxPool2d(2, 2)
        self.v_down1 = ViT(image_size=(224, 224), patch_size=(16, 16), num_classes=64, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 64, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=112)
        self.v_down2 = ViT(image_size=(112, 112), patch_size=(8, 8), num_classes=128, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 128, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=56)
        self.v_down3 = ViT(image_size=(56, 56), patch_size=(4, 4), num_classes=256, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 256, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=28)


        self.vconv1 = DoubleConv(3, 64)
        self.vconv2 = DoubleConv(64, 128)
        self.vconv3 = DoubleConv(128, 256)

        self.e_model2_1 = CMT(64)
        self.e_model2_2 = CMT(128)
        self.e_model2_3 = CMT(256)

        self.cvit1 = ViT(image_size=(224, 224), patch_size=(16, 16), num_classes=64, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 3, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=224)
        self.cvit2 = ViT(image_size=(112, 112), patch_size=(8, 8), num_classes=128, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 64, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=112)
        self.cvit3 = ViT(image_size=(56, 56), patch_size=(4, 4), num_classes=256, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 128, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=56)

        self.cv_skips1 = CV_skips(channel=64)
        self.cv_skips2 = CV_skips(channel=128)
        self.cv_skips3 = CV_skips(channel=256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uconv1 = DoubleConv(768, 256)
        self.uconv2 = DoubleConv(384, 128)
        self.uconv3 = DoubleConv(192, 64)

        self.finalConv = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv64 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv128 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv256 = nn.Conv2d(128, 256, kernel_size=1)

        resnet = resnet_model.resnet34(pretrained=True)

        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
#--------------------------------------------------------------------------------------------------------------------------
        self.conv_3 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_3_cv = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.Max_pool = nn.MaxPool2d(2,2)
        self.conv_1 = nn.Conv2d(1024, 1024, 1, 1, bias=False)
        self.conv_1_2 = nn.Conv2d(512, 1024, 1, 1, bias=False)
        self.conv_1_3 = nn.Conv2d(256, 512, 1, 1, bias=False)
        self.conv_1_4 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.conv_1_5 = nn.Conv2d(1024, 512, 1, 1, bias=False)
        self.conv_1_6 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_dS = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_dS_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_dS_3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, groups=256, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.grouped_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, groups=2)
        self.gaussian_kernel = self.create_gaussian_kernel(5, 1.0).to('cuda:0')
#------------------------------------------------------------------------------------------------------------------
    def create_gaussian_kernel(self, size, sigma):
        x = torch.arange(size).float().to('cuda:0')
        y = torch.arange(size).float().to('cuda:0')
        x_grid, y_grid = torch.meshgrid(x, y)

        kernel = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(
            -((x_grid - size // 2) ** 2 + (y_grid - size // 2) ** 2) / (2 * sigma ** 2)
        )
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):

        # resnet
        e0 = self.first_conv(x)
        e0 = self.first_bn(e0)
        res_e0 = self.first_relu(e0)    #
        res_e1 = self.encoder1(res_e0)  # 64,112
        res_e2 = self.encoder2(res_e1)  # 128,56
        res_e3 = self.encoder3(res_e2)  # 256,28

        x_c = x
        x_v = x
        x_c1 = self.cvit1(x_c)
        x_c_1 = self.conv64(x_c)
        x_c1 = x_c1 + x_c_1
        x_c1 = self.relu(x_c1)
        x_v1 = self.vconv1(x_v)
        x_v1 = self.e_model2_1(x_v1)
        x_c1, x_v1_1, skips1 = self.cv_skips1(x_c1, x_v1)
        x_c1 = self.c_down(x_c1)
        x_v1 = self.v_down1(x_v1_1)
        x_v1 = x_v1 + res_e1
        x_v1 = self.relu(x_v1)

        x_c2 = self.cvit2(x_c1) # c1 64 112 112
        x_c1_1 = self.conv128(x_c1)
        x_c2 = x_c2 + x_c1_1
        x_c2 = self.relu(x_c2)
        x_v2 = self.vconv2(x_v1)
        x_v2 = self.e_model2_2(x_v2)
        x_c2, x_v2_1, skips2 = self.cv_skips2(x_c2, x_v2)
        x_c2 = self.c_down(x_c2)
        x_v2 = self.v_down2(x_v2_1)
        x_v2 = x_v2 + res_e2
        x_v2 = self.relu(x_v2)

        x_c3 = self.cvit3(x_c2)
        x_c2_1 = self.conv256(x_c2)
        x_c3 = x_c3 + x_c2_1
        x_c3 = self.relu(x_c3)
        x_v3 = self.vconv3(x_v2)
        x_v3 = self.e_model2_3(x_v3)
        x_c3, x_v3_1, skips3 = self.cv_skips3(x_c3, x_v3)
        x_c3 = self.c_down(x_c3)
        x_v3 = self.v_down3(x_v3_1)
        x_v3 = x_v3 + res_e3
        x_v3 = self.relu(x_v3)
#--------------------------------------------------------------------------------------------------------------
        # bottleneck
        x_bot = torch.cat((x_c3, x_v3), dim=1)
        x_bot = self.bottleneck(x_bot)
        x_skips3 = self.conv_1_3(skips3)
        gaussian_kernel_expanded = self.gaussian_kernel.expand(x_skips3.shape[1], 1, 5, 5)
        x_skips3 = F.conv2d(x_skips3, gaussian_kernel_expanded, padding=2, groups=x_skips3.shape[1])
        x_skips3 = self.conv_2(x_skips3)
        x = torch.cat((x_skips3,x_bot),dim=1)
        x1 = self.conv_3(x)
        x2 = self.conv_dS(x)
        x3 = self.avg_pool(x1+x2)
        x4 = self.max_pool(x1+x2)
        x5 = self.conv_1(x3)
        x6 = self.conv_1(x4)
        x7 = self.sigmoid(x5)
        x8 = self.sigmoid(x6)
        x9 = (x7+x8)/2
        x = x9 * x
        x_skips3 = self.conv_1_2(x_skips3)
        x_bo = self.conv_1_2(x_bot)
        x_sk = x9 * x_skips3
        x_bo = (1-x9)*x_bo
        x_loss = torch.abs(x_sk - x_bo)
        x = x + x_loss
        x1 = self.conv_3(x)
        x2 = self.conv_dS(x)
        x3 = self.avg_pool(x1 + x2)
        x4 = self.max_pool(x1 + x2)
        x5 = self.conv_1(x3)
        x6 = self.conv_1(x4)
        x7 = self.sigmoid(x5)
        x8 = self.sigmoid(x6)
        x9 = (x7 + x8) / 2
        x = x9 * x
        x_bot = x + x_bo + x_skips3
        x_bot = self.conv_1_5(x_bot)
#--------------------------------------------------------------------------------------------------------------
        x_up1 = self.up(x_bot) # 512 28 28 -> 512 56 56
        skips3 = x_v3_1 + skips3
        x_up1 = torch.cat((x_up1, skips3), dim=1) # 758 56 56
        x_up1 = self.uconv1(x_up1) # 256 56 56

        x_up2 = self.up(x_up1)
        skips2 = x_v2_1 + skips2
        x_up2 = torch.cat((x_up2, skips2), dim=1)
        x_up2 = self.uconv2(x_up2)

        x_up3 = self.up(x_up2)
        skips1 = x_v1_1 + skips1
        x_up3 = torch.cat((x_up3, skips1), dim=1)
        x_up3 = self.uconv3(x_up3)

        x_out = self.finalConv(x_up3)

        return x_out
