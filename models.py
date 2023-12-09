import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block


def get_model(args, n_modality, input_shape, patch_size):
    if args.model == 'mae':
        return MAE(n_modality=n_modality,
                   input_shape=input_shape, patch_size=patch_size,
                   embed_dim_enc=args.embed_dim_enc, embed_dim_dec=args.embed_dim_dec,
                   num_layer_enc=args.num_layer_enc, num_layer_dec=args.num_layer_dec,
                   num_head_enc=args.num_head_enc, num_head_dec=args.num_head_dec,
                   dropout=args.dropout, mask_ratio=args.mask_ratio)
    elif args.model == 'metamae':
        return MetaMAE(n_modality=n_modality,
                       input_shape=input_shape, patch_size=patch_size,
                       embed_dim_enc=args.embed_dim_enc, embed_dim_dec=args.embed_dim_dec,
                       num_layer_enc=args.num_layer_enc, num_layer_dec=args.num_layer_dec,
                       num_head_enc=args.num_head_enc, num_head_dec=args.num_head_dec,
                       dropout=args.dropout, mask_ratio=args.mask_ratio,
                       inner_lr=args.inner_lr, reg_weight=args.reg_weight, s_ratio=args.s_ratio,
                       use_first_order=args.use_first_order)
    else:
        raise NotImplementedError


class MAE(nn.Module):
    def __init__(self,
                 n_modality: int = 1,
                 input_shape: tuple[int] = (3, 224, 224),
                 patch_size: tuple[int] = (16, 16),
                 embed_dim_enc: int = 1024,
                 embed_dim_dec: int = 512,
                 num_layer_enc: int = 24,
                 num_layer_dec: int = 8,
                 num_head_enc: int = 16,
                 num_head_dec: int = 16,
                 dropout: float = 0.,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_shape = input_shape
        #encoder
        self.patch_embed = UnifiedPatchEmbed(n_modality=n_modality, input_shape=input_shape, patch_size=patch_size, embed_dim=embed_dim_enc)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim_enc))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim_enc), requires_grad=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.encoder = nn.Sequential(*[Block(embed_dim_enc, num_head_enc, drop=dropout, mlp_ratio=2) for _ in range(num_layer_enc)])
        self.norm_enc = nn.LayerNorm(embed_dim_enc)

        #decoder
        self.decoder_embed = nn.Linear(embed_dim_enc, embed_dim_dec, bias=True)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, embed_dim_dec))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim_dec), requires_grad=False)

        self.decoder = nn.Sequential(*[Block(embed_dim_dec, num_head_dec, mlp_ratio=2) for _ in range(num_layer_dec)])
        self.norm_dec = nn.LayerNorm(embed_dim_dec)
        if self.patch_embed.n_dim == 1:
            if self.patch_embed.is_tokenize_data:
                self.decoder_head = nn.Linear(embed_dim_dec, embed_dim_enc, bias=True)
            else:
                self.decoder_head = nn.Linear(embed_dim_dec, input_shape[0]*patch_size[0], bias=True)
        elif self.patch_embed.n_dim == 2:
            self.decoder_head = nn.Linear(embed_dim_dec, input_shape[0]*patch_size[0]*patch_size[1], bias=True)
        else:
            raise NotImplementedError

        self.initialize_weights()

    def initialize_weights(self):
        if self.patch_embed.n_dim == 1:
            pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        elif self.patch_embed.n_dim == 2:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.patch_embed.is_tokenize_data:
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w)
        else:
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch, mode='train', **kwargs):
        if mode == 'train':
            return self.compute_loss(batch)
        elif mode == 'feature':
            return self.extract_features(batch, **kwargs)

    def compute_loss(self, batch):
        x, _ = batch
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred   = self.forward_decoder(latent, ids_restore)
        target = self.patchify(x)
        loss   = self.get_recon_loss(pred, target, mask)
        return dict(loss=loss)

    def extract_features(self, batch, eval='global_pool'):
        x = self.patch_embed(batch)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        if eval == 'global_pool':
            x = self.norm_enc(x)[:, 1:].mean(dim=1)
        elif eval == 'tokenize':
            x = self.norm_enc(x)[:, 1:]
        else:
            raise NotImplementedError
        return x

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        if self.patch_embed.is_tokenize_data:
            return imgs
        elif self.patch_embed.n_dim == 1:
            assert imgs.shape[2] % p == 0
            h = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], self.input_shape[0], h, p))
            x = torch.einsum('nchp->nhpc', x)
            x = x.reshape(shape=(imgs.shape[0], h, p*self.input_shape[0]))
        elif self.patch_embed.n_dim == 2:
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], self.input_shape[0], h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.input_shape[0]))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        if self.patch_embed.n_dim == 1:
            h == int(x.shape[1]**.5)
            x = x.reshape(shape=(x.shape[0], h, p, self.input_shape[0]))
            x = torch.einsum('nhpc->nchp', x)
            imgs = x.reshape(shape=(x.shape[0], self.input_shape[0], h*p))
            return imgs
        elif self.patch_embed.n_dim == 2:
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]

            x = x.reshape(shape=(x.shape[0], h, w, p, p, self.input_shape[0]))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], self.input_shape[0], h * p, h * p))
            return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm_enc(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.norm_dec(x)
        x = self.decoder_head(x)
        x = x[:, 1:, :]
        if self.patch_embed.is_tokenize_data:
            x = torch.einsum('nhe,ve->nhv', x, self.patch_embed.proj.weight).transpose(1, 2)
        return x

    def get_recon_loss(self, pred, target, mask):
        if self.patch_embed.is_tokenize_data:
            loss = F.cross_entropy(pred, target, reduction='none')
        else:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class MetaMAE(MAE):
    def __init__(self,
                 n_modality: int = 1,
                 input_shape: tuple[int] = (3, 224, 224),
                 patch_size: int = 16,
                 embed_dim_enc: int = 1024,
                 embed_dim_dec: int = 512,
                 num_layer_enc: int = 24,
                 num_layer_dec: int = 8,
                 num_head_enc: int = 16,
                 num_head_dec: int = 16,
                 dropout: float = 0.,
                 mask_ratio: float = 0.75,
                 inner_lr: float = 0.1,
                 reg_weight: float = 1,
                 s_ratio: float = 0.1,
                 use_first_order: bool = False):
        super().__init__(n_modality, input_shape, patch_size,
                         embed_dim_enc, embed_dim_dec,
                         num_layer_enc, num_layer_dec,
                         num_head_enc, num_head_dec,
                         dropout, mask_ratio)
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.reg_weight = reg_weight
        self.s_ratio = s_ratio
        self.use_first_order = use_first_order
        self.inner_lr = nn.Parameter(torch.tensor(inner_lr, dtype=torch.float), requires_grad=False)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim_enc, embed_dim_enc*4, bias=False),
            nn.BatchNorm1d(embed_dim_enc*4),
            nn.ReLU(),
            nn.Linear(embed_dim_enc*4, 128, bias=False)
        )

    def get_nearby_s_mask(self, mask):
        unmasked = 1 - mask
        if len(self.input_shape) == 2:
            kernel = torch.tensor([1, 1, 1], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
            padded_masked = F.pad(unmasked.unsqueeze(1).float(), (1, 1), mode='constant', value=0)
            nearby_masks = F.conv1d(padded_masked, kernel, stride=1, padding=0).squeeze(1)
            nearby_masks = nearby_masks.logical_or(unmasked).reshape(mask.shape[0], -1)
        elif len(self.input_shape) == 3:
            unmasked_section = unmasked.reshape(mask.shape[0], self.input_shape[1]//self.patch_size[0], self.input_shape[2]//self.patch_size[1])
            kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
            padded_masked = F.pad(unmasked_section.unsqueeze(1).float(), (1, 1, 1, 1), mode='constant', value=0)
            nearby_masks = F.conv2d(padded_masked, kernel, stride=1, padding=0).squeeze(1)
            nearby_masks = nearby_masks.logical_or(unmasked_section).reshape(mask.shape[0], -1)

        return nearby_masks

    def inner_loop_update(self, latent, mask_support, ids_restore, target):
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.get_recon_loss(pred, target, mask_support)
        grad, = torch.autograd.grad(loss*pred.shape[0], inputs=latent, create_graph=not self.use_first_order)
        latent = latent - self.inner_lr * grad

        return latent

    def forward_loss_meta(self, latent, mask, ids_restore, target):
        unmasked = 1 - mask
        prev_latent = latent

        latent.retain_grad()

        mask_support = mask.mul(self.get_nearby_s_mask(mask)) * (torch.rand(mask.shape, device=mask.device) < self.s_ratio) + unmasked
        latent = self.inner_loop_update(latent, mask_support, ids_restore, target)

        #outer loop
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.get_recon_loss(pred, target, mask)

        #contrast
        z1 = F.normalize(self.projector(prev_latent[:, 1:].mean(dim=1)))
        z2 = F.normalize(self.projector(latent[:, 1:].mean(dim=1)))
        z = torch.cat([z1, z2])
        logits = torch.mm(z, z.T).div(0.5)
        logits.fill_diagonal_(float('-inf'))
        labels = torch.tensor(list(range(z1.shape[0], 2*z1.shape[0])) + list(range(z1.shape[0])), device=logits.device)
        loss += self.reg_weight * F.cross_entropy(logits, labels)

        return loss

    def compute_loss(self, batch, **kwargs):
        x, _ = batch
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        target = self.patchify(x)
        loss = self.forward_loss_meta(latent, mask, ids_restore, target)
        return dict(loss=loss)


class UnifiedPatchEmbed(nn.Module):
    def __init__(self,
                 n_modality: int = 1,
                 input_shape: tuple[int] = (3, 224, 224),
                 patch_size: tuple[int] = (16, 16),
                 embed_dim: int = 192):
        super().__init__()
        self.n_modality = n_modality
        if self.n_modality == 1:
            self.n_dim = len(input_shape) - 1
            self.is_tokenize_data = False

            if self.n_dim == 1: #input_shape: channels, seq_len or (vocab_len), seq_len
                self.patch_size = patch_size
                self.grid_size = (input_shape[1] // patch_size[0],)
                self.num_patches = self.grid_size[0]
                if isinstance(input_shape[0], tuple): #token data
                    self.proj = nn.Embedding(input_shape[0][0], embed_dim)
                    self.is_tokenize_data = True
                else:
                    self.proj = nn.Conv1d(input_shape[0], embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
            elif self.n_dim == 2:
                self.patch_size = patch_size
                self.grid_size = (input_shape[1] // patch_size[0], input_shape[2] // patch_size[1])
                self.num_patches = self.grid_size[0] * self.grid_size[1]
                self.proj = nn.Conv2d(input_shape[0], embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
            else:
                raise NotImplementedError
        else: #TODO. Now, we do not support vision-language in this code
            raise NotImplementedError

    def forward(self, x):
        if self.n_modality == 1:
            x = self.proj(x)
            if self.n_dim == 2:
                x = x.flatten(2)
            if not self.is_tokenize_data:
                x = x.transpose(1, 2)
            return x
        else:
            raise NotImplementedError


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32).reshape(1, 1, grid_size)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

