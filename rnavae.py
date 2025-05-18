import os, math
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from itertools import pairwise
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset

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
    dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
    self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
    self.output_proj = nn.Linear(d_model, 3)
  
  def forward(self, z, seq_len):
    memory = self.latent_to_mem(z).unsqueeze(1)
    tgt = self.tgt_token.expand(z.size(0), seq_len, -1)
    tgt = self.pos_enc(tgt)
    h = self.decoder(tgt=tgt, memory=memory)
    return self.output_proj(h)

class RNA3D_VAE(nn.Module):
  def __init__(self, d_model=128, num_layers=4, latent_dim=32):
    super().__init__()
    self.encoder = RNA3d_Encoder(d_model, num_layers, latent_dim)
    self.decoder = RNA3d_Decoder(d_model, num_layers, latent_dim)

  def encode(self, x): return self.encoder(x)
  def decode(self, z, seq_len): return self.decoder(z, seq_len)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z, x.size(1))
    return recon, mu, logvar
  
batch_size = 32
max_len = 384
min_length_filter = 10
max_length_filter = 9999999
cutoff_date = pd.Timestamp("2020-01-01")
test_cutoff_date = pd.Timestamp("2022-05-01")
train_sequences_path = "stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "stanford-rna-3d-folding/train_labels.csv"
pretrained_weights_path = "ribonanzanet-weights/RibonanzaNet.pt"

data = dataset.load_data(train_sequences_path, train_labels_path, min_length_filter, max_length_filter)
train_indices, test_indices = dataset.train_val_split(data, cutoff_date, test_cutoff_date)

train_dataset = dataset.RNA3D_Dataset(train_indices, data, max_len=max_len)
val_dataset = dataset.RNA3D_Dataset(test_indices, data, max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def vae_loss(recon_x, x, recon_sigma, target_sigma, mask, mu, logvar, epoch, sigma_weight=0.01, kl_weight=1):
  mse = ((recon_x - x) ** 2).sum(-1)   # [B,L]
  recon_loss = ((mse * mask).sum(dim=1) / mask.sum(dim=1)).mean()
  sigma_loss = torch.mean((recon_sigma - target_sigma) ** 2)
  kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
  kl_weight = min(epoch / 250, 1.0) * kl_weight
  return recon_loss + sigma_weight * sigma_loss + kl_weight * kl_loss, recon_loss, sigma_loss, kl_loss

def train_epoch(model, loader, optimizer, device, accum_steps, epoch):
  model.train()
  running_loss = 0.0
  optimizer.zero_grad()

  for step, batch, in enumerate(loader):
    x = batch["xyz"].to(device)
    mask = batch["mask"].to(device)
    sigma = batch["sigma"].to(device)
    recon, mu, logvar = model(x)
    recon_sigma = recon[:, 0, 0]
    # drop the last token (scale sigma)
    recon_coords = recon[:, 1:]
    target_coords = x[:, 1:]
    mask_coords = mask[:, 1:]
    loss, _, _, _ = vae_loss(recon_coords, target_coords, recon_sigma, sigma, mask_coords, mu, logvar, epoch)
    loss = loss / accum_steps
    loss.backward()

    if (step + 1) % accum_steps == 0:
      optimizer.step()
      optimizer.zero_grad()

    running_loss += (loss.item() * accum_steps)
  
  return running_loss / len(loader)

def valid_loss(model, loader, device, epoch):
  model.eval()
  tot_loss = tot_recon_l = tot_sigma_l = tot_kl_l = 0.0
  ns = len(loader.dataset)
  with torch.no_grad():
    for batch in loader:
      x = batch["xyz"].to(device)
      mask = batch["mask"].to(device)
      sigma = batch["sigma"].to(device)
      recon, mu, logvar = model(x)
      recon_sigma = recon[:, 0, 0]
      # drop the last token (scale sigma)
      recon_coords = recon[:, 1:]
      target_coords = x[:, 1:]
      mask_coords = mask[:, 1:]
      B = x.size(0)
      loss, recon_l, sigma_l, kl_l = vae_loss(recon_coords, target_coords, recon_sigma, sigma, mask_coords, mu, logvar, epoch)
      tot_loss += loss.item() * B
      tot_recon_l += recon_l.item() * B
      tot_sigma_l += sigma_l.item() * B
      tot_kl_l += kl_l.item() * B
  return tot_loss / ns, tot_recon_l / ns, tot_sigma_l / ns, tot_kl_l / ns

device = torch.device(
  "mps"  if torch.backends.mps.is_available() else
  "cuda" if torch.cuda.is_available() else
  "cpu")
d_model = 128
num_layers = 3
latent_dim = 64
lr = 1e-4
accum_steps = 4
num_epochs = 5000

model = RNA3D_VAE(d_model, num_layers, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

import matplotlib as mpl
mpl.rcParams["figure.raise_window"] = False
import matplotlib.pyplot as plt
plt.ion()
train_curve, val_curve = [], []
val_rec_curve, val_kl_curve = [], []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
ax1.set_title("Total loss");
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax2.set_title("Validation components")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss")
train_line, = ax1.plot([], [], label="train")
val_line,   = ax1.plot([], [], label="val")
rec_line,   = ax2.plot([], [], label="val recon")
kl_line,    = ax2.plot([], [], label="val KL")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.pause(0.01)

model.eval()
_, init_train, _, _ = valid_loss(model, train_loader, device, epoch=0)
_, init_val, _, _ = valid_loss(model, val_loader, device, epoch=0)
print("untrained  train:", init_train, "  val:", init_val)
def coord_var(loader):
  all_var = []
  for b in loader:
    xyz = b["xyz"]              # [B,L,3]  already centred
    mask = b["mask"]
    # squared distance from origin
    sq = (xyz ** 2).sum(-1)     # [B,L]
    var = (sq * mask).sum() / mask.sum()   # average per residue
    all_var.append(var.item())
  return sum(all_var) / len(all_var)
print("train  coord variance ≈", coord_var(train_loader))
print("valid  coord variance ≈", coord_var(val_loader))

for epoch in range(1, num_epochs + 1):
  train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps, epoch)
  val_loss, val_rec, val_sigma, val_kl = valid_loss(model, val_loader, device, epoch)
  
  train_curve.append(train_loss)
  val_curve.append(val_loss)
  val_rec_curve.append(val_rec)
  val_kl_curve.append(val_kl)
  epochs = range(1, epoch + 1)
  train_line.set_data(epochs, train_curve)
  val_line.set_data(epochs, val_curve)
  rec_line.set_data(epochs, val_rec_curve)
  kl_line.set_data(epochs, val_kl_curve)
  ax1.relim(); ax1.autoscale_view()
  ax2.relim(); ax2.autoscale_view()
  plt.pause(0.01)   # brief GUI event flush

  print(f"Epoch {epoch:3d}/{num_epochs}  train={train_loss:.3f}  val={val_loss:.3f}  val_sigma={val_sigma:.3f}  val_rec={val_rec:.3f}  val_kl={val_kl:.3f}")

torch.save(model.state_dict(), "rna3d_vae.pt")

# TODO: this is currently on per-residue loss
# switch to per-molecule (or hybrid)
#√L weight
#Multiply each sample’s loss by 1/\sqrt{L}.  Longer sequences still matter more, but not quadratically.
#Curriculum
#Start with per-residue loss for stability; after N epochs switch to per-molecule to fine-tune global shape.
#Task-weighted mix
#Total loss = \alpha\,L_\text{per-res} + (1-\alpha)\,L_\text{per-mol}.  Pick α≈0.7 so both signals are present
