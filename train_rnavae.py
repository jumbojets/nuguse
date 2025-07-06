import os, time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
from rnavae import RNA3d_VQVAE, vae_loss

torch.set_float32_matmul_precision('medium')

batch_size = 16
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

train_dataset = dataset.RNA3D_Dataset(train_indices, data, max_len=max_len, rotate=False)
val_dataset = dataset.RNA3D_Dataset(test_indices, data, max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# overfit a single batch
# from torch.utils.data import Subset
# subset = Subset(train_dataset, list(range(batch_size)))
# train_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

@torch.compile
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if (step + 1) % accum_steps == 0:
      optimizer.step()
      optimizer.zero_grad()

    running_loss += (loss.item() * accum_steps)
  
  return running_loss / len(loader)

@torch.compile
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
d_model = 256
num_layers = 5
latent_dim = 64
lr = 1e-4
accum_steps = 1
num_epochs = 5000
checkpoint_every = 500
# load = "rna3d_vae_0500-5000.pt"
load = None

model = RNA3d_VQVAE(d_model, num_layers, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

if load is not None: model.load_state_dict(torch.load(load, weights_only=True))

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
  st = time.monotonic()
  train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps, epoch)
  val_loss, val_rec, val_sigma, val_kl = valid_loss(model, val_loader, device, epoch)
  elapsed = time.monotonic() - st
  
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
  plt.savefig("train.png")

  print(f"Epoch {epoch:3d}/{num_epochs}  {train_loss=:.3f}  {val_loss=:.3f}  {val_sigma=:.3f}  {val_rec=:.3f}  {val_kl=:.3f}  {elapsed=:.3f}s")

  if epoch == num_epochs or epoch % checkpoint_every == 0: torch.save(model.state_dict(), f"rna3d_vae_{epoch:04d}-{num_epochs}.pt")

torch.save(model.state_dict(), "rna3d_vae.pt")

# TODO: this is currently on per-residue loss
# switch to per-molecule (or hybrid)
#√L weight
#Multiply each sample’s loss by 1/\sqrt{L}.  Longer sequences still matter more, but not quadratically.
#Curriculum
#Start with per-residue loss for stability; after N epochs switch to per-molecule to fine-tune global shape.
#Task-weighted mix
#Total loss = \alpha\,L_\text{per-res} + (1-\alpha)\,L_\text{per-mol}.  Pick α≈0.7 so both signals are present


# VQ-VAE
# REMUL (https://arxiv.org/html/2410.17878v1#S1)
# GAN critic loss
# perception loss
# Aux decoder (https://sander.ai/images/aux_decoder.png)

