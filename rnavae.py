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

class VectorQuantizer(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

  def forward(self, inputs):
    # inputs: (B, L, D)
    input_shape = inputs.shape
    flat_input = inputs.view(-1, self.embedding_dim)  # (B*L, D)

    distances = (
      torch.sum(flat_input**2, dim=1, keepdim=True)
      + torch.sum(self.embedding.weight**2, dim=1)
      - 2 * torch.matmul(flat_input, self.embedding.weight.t())
    )  # (B*L, num_embeddings)

    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*L, 1)
    encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)  # one-hot (B*L, num_embeddings)

    quantized = torch.matmul(encodings, self.embedding.weight)  # (B*L, D)
    quantized = quantized.view(input_shape)  # (B, L, D)

    # Straight-through estimator
    quantized = inputs + (quantized - inputs).detach()

    # Compute losses
    e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
    q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    return quantized, loss, perplexity

class RNA3d_Encoder(nn.Module):
  def __init__(self, d_model, num_layers, latent_dim):
    super().__init__()
    self.input_proj = nn.Linear(3, d_model)
    self.pos_enc = PositionalEncoding(d_model)
    enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.fc = nn.Linear(d_model, latent_dim)

  def forward(self, x):
    h = self.input_proj(x)
    h = self.pos_enc(h)
    h = self.encoder(h)
    z_e = self.fc(h)  # (B, L, latent_dim)
    return z_e

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

class RNA3d_VQVAE(nn.Module):
  def __init__(self, d_model=128, num_layers=4, latent_dim=32, num_embeddings=512, commitment_cost=0.25):
    super().__init__()
    self.encoder = RNA3d_Encoder(d_model, num_layers, latent_dim)
    self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
    self.decoder = RNA3d_Decoder(d_model, num_layers, latent_dim)

  def encode(self, x):
    z_e = self.encoder(x).mean(dim=1)  # Pool over length
    z_q, vq_loss, perplexity = self.vq_layer(z_e.unsqueeze(1))
    z_q = z_q.squeeze(1)
    return z_q, vq_loss, perplexity

  def decode(self, z, seq_len): return self.decoder(z, seq_len)

  def forward(self, x):
    z_q, vq_loss, perplexity = self.encode(x)
    recon = self.decode(z_q, x.size(1))
    return recon, vq_loss, perplexity

def vae_loss(recon_x, x, recon_sigma, target_sigma, mask, vq_loss, global_step, sigma_weight=0.1, kl_weight=0.1):
  mse = ((recon_x - x) ** 2).sum(-1)   # [B,L]
  recon_loss = ((mse * mask).sum(dim=1) / mask.sum(dim=1)).mean()
  sigma_loss = torch.mean((recon_sigma - target_sigma) ** 2)
  return recon_loss + sigma_weight * sigma_loss + kl_weight * vq_loss, recon_loss, sigma_loss, vq_loss
