import os, math
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=4096):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]
  
  def forward(self, x): return x + self.pe[:, : x.size(1)]

class RNA3d_Encoder(nn.Module):
  def __init__(self, d_model, num_layers, latent_dim):
    super().__init__()
    self.input_proj = nn.Linear(3, d_model)
    self.pos_enc = PositionalEncoding(d_model)
    enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.fc_mu = nn.Linear(d_model, latent_dim)
    self.fc_logv = nn.Linear(d_model, latent_dim)

  def forward(self, x):
    h = self.input_proj(x)
    h = self.pos_enc(h)
    h = self.encoder(h)
    pooled = h.mean(dim=1)
    return self.fc_mu(pooled), self.fc_logv(pooled).clamp(-10.0, 10.0)

class RNA3d_Decoder(nn.Module):
  def __init__(self, d_model, num_layers, latent_dim):
    super().__init__()
    self.latent_to_mem = nn.Linear(latent_dim, d_model)
    self.pos_enc = PositionalEncoding(d_model)
    self.tgt_token = nn.Parameter(torch.zeros(1, 1, d_model))
    dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*1, dropout=0.1, batch_first=True)
    self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
    self.output_proj = nn.Linear(d_model, 3)

  def forward(self, z, seq_len):
    memory = self.latent_to_mem(z).unsqueeze(1).repeat(1, seq_len, 1)
    memory = self.pos_enc(memory)
    tgt = self.tgt_token.expand(z.size(0), seq_len, -1)
    tgt = self.pos_enc(tgt)
    h = self.decoder(tgt=tgt, memory=memory)
    return self.output_proj(h)

class RNA3d_VAE(nn.Module):
  def __init__(self, d_model=128, num_layers=4, latent_dim=32):
    super().__init__()
    self.encoder = RNA3d_Encoder(d_model, num_layers, latent_dim)
    self.decoder = RNA3d_Decoder(d_model, num_layers, latent_dim)

  def encode(self, x): return self.encoder(x)
  def decode(self, z, seq_len): return self.decoder(z, seq_len)

  @staticmethod
  def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z, x.size(1))
    return recon, mu, logvar
  
def kl_with_free_bits(mu, logvar, global_step, bits=0.05):
  kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
  kl = torch.clamp(kl, min=bits)
  beta = min(global_step / 2000, 1.0)
  return beta * kl.mean(), kl.mean()

def vae_loss(recon_x, x, recon_sigma, target_sigma, mask, mu, logvar, global_step, sigma_weight=0.1, kl_weight=1):
  mse = ((recon_x - x) ** 2).sum(-1)   # [B,L]
  recon_loss = ((mse * mask).sum(dim=1) / mask.sum(dim=1)).mean()
  sigma_loss = torch.mean((recon_sigma - target_sigma) ** 2)
  kl_beta, kl_loss = kl_with_free_bits(mu, logvar, global_step)
  return recon_loss + sigma_weight * sigma_loss + kl_beta, recon_loss, sigma_loss, kl_loss
