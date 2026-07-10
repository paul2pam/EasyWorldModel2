import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "rssm")
from rssm import RSSM
from dataset import SequenceDataset

# --- Data ---
HDF5_PATH = "can_ph_image.hdf5"

# --- Dims ---
EMBED_DIM    = 512
HIDDEN_DIM   = 512
DETER_DIM    = 512
DISCRETE_DIM = 1024  # 32 categoricals x 32 classes
ACTION_DIM   = 7   # robomimic Lift: [3 pos + 3 rot + 1 gripper]

# --- Training ---
SEQ_LEN    = 50   # consecutive frames per training sequence
BATCH_SIZE = 16
LR         = 1e-4
KL_WEIGHT  = 0.1
FREE_BITS  = 1.0  # minimum KL per step — prevents posterior collapse
TRAIN_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = RSSM(EMBED_DIM, HIDDEN_DIM, DETER_DIM, DISCRETE_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

dataset     = SequenceDataset(HDF5_PATH, seq_len=SEQ_LEN)
loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
loader_iter = iter(loader)


def train_step():
    global loader_iter
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    obs = batch['image'].to(device)   # (B, T, 3, 64, 64)
    act = batch['action'].to(device)  # (B, T, 7)

    h, z = model.initial(BATCH_SIZE)
    recon_loss = 0.0
    kl = 0.0

    for t in range(SEQ_LEN):
        obs_t = obs[:, t]  # (B, 3, 64, 64)
        act_t = act[:, t]  # (B, 7)

        e = model.encoder(obs_t)
        h, z, prior_logits, post_logits = model.observation_step(h, z, act_t, e)

        recon = model.decode(h, z)
        recon_loss += F.mse_loss(recon, obs_t)
        kl += model.kl_loss(prior_logits, post_logits, FREE_BITS)

    recon_loss = recon_loss / SEQ_LEN
    kl = kl / SEQ_LEN
    total_loss = recon_loss + KL_WEIGHT * kl
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    return recon_loss.item(), kl.item()


print(f"Dataset: {len(dataset)} windows from {HDF5_PATH}")
print(f"\nTraining for {TRAIN_STEPS} steps...")
for step in range(TRAIN_STEPS):
    recon_loss, kl = train_step()

    if step % 50 == 0:
        print(f"Step {step:4d} | recon: {recon_loss:.5f} | kl: {kl:.5f}")

    if step % 500 == 0 and step > 0:
        torch.save(model.state_dict(), f"checkpoint_{step}.pt")
        print(f"  Saved checkpoint_{step}.pt")

torch.save(model.state_dict(), "checkpoint.pt")
print("Saved checkpoint.pt")
print("Done.")
