"""
honet/distiller.py
------------------
Implements the knowledge distillation step that compresses a trained Octave's
learned representation of a task into a compact "Master-Tone" vector I.

The Master-Tone acts as a summary of everything the HONet has learned up to
the current task.  When a new Octave is trained, it is conditioned on this
vector, enabling positive forward knowledge transfer without any risk of
overwriting previously frozen skills.

Algorithm
---------
1. Freeze the newly trained Octave (its weights are immutable from this
   point on).
2. Train a lightweight MasterToneProducer network to predict a single
   representative embedding vector from a batch of task data.
3. Average the producer's output across all training batches to obtain a
   stable, deterministic Master-Tone I_new.
4. Return I_new (detached from the computation graph).
"""

import torch
import torch.nn as nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# MasterToneProducer
# ---------------------------------------------------------------------------

class MasterToneProducer(nn.Module):
    """
    A small network that encodes a *batch* of samples into a single
    representative Master-Tone vector via mean-pooling over the batch.

    It learns to produce a conditioning vector that best summarises the
    statistical structure of the task data, as judged by how well it lets
    a student network mimic the frozen Octave's latent distribution.

    Parameters
    ----------
    master_tone_dim: target dimensionality of the output Master-Tone
    input_dim      : dimensionality of a single (flattened) sample
    hidden_dim     : MLP hidden width
    """

    def __init__(self, master_tone_dim: int, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, master_tone_dim),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_flat : (batch, input_dim) – pre-flattened samples

        Returns
        -------
        master_tone : (master_tone_dim,) – batch-mean embedding
        """
        embeddings = self.net(x_flat)          # (batch, master_tone_dim)
        return embeddings.mean(dim=0)           # (master_tone_dim,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distill_knowledge_to_master_tone(
    octave: nn.Module,
    data_loader,
    previous_master_tone: torch.Tensor | None,
    device: torch.device,
    master_tone_dim: int,
    z_dim: int,
    num_epochs: int = 5,
    lr: float = 5e-4,
) -> torch.Tensor:
    """
    Distil a trained (frozen) Octave into a Master-Tone vector.

    Parameters
    ----------
    octave               : the trained, frozen Octave module
    data_loader          : DataLoader for the current task's training data
    previous_master_tone : Master-Tone from the previous task (None for task 1)
    device               : torch device
    master_tone_dim      : output dimensionality of the Master-Tone
    z_dim                : latent dimensionality of the Octave
    num_epochs           : number of distillation epochs
    lr                   : learning rate for the producer

    Returns
    -------
    I_new : (master_tone_dim,) tensor – the distilled Master-Tone (detached)
    """
    # Freeze Octave permanently
    octave.eval()
    for p in octave.parameters():
        p.requires_grad_(False)

    # Determine flat input dimension from first batch
    sample_x, _ = next(iter(data_loader))
    sample_x = sample_x.to(device)
    flat_dim = sample_x.view(sample_x.size(0), -1).shape[1]

    producer = MasterToneProducer(master_tone_dim, flat_dim).to(device)
    optimizer = Adam(producer.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            x_flat = x_batch.view(x_batch.size(0), -1)

            # Candidate Master-Tone from this batch
            I_candidate = producer(x_flat)                          # (master_tone_dim,)
            I_cond = I_candidate.unsqueeze(0).expand(x_batch.size(0), -1)

            # Target: mu from the frozen Octave
            with torch.no_grad():
                if previous_master_tone is not None:
                    prev_cond = previous_master_tone.unsqueeze(0).expand(x_batch.size(0), -1)
                else:
                    prev_cond = None
                _, mu_target, _ = octave(x_batch, prev_cond)

            # Predicted: mu from Octave conditioned on I_candidate
            _, mu_pred, _ = octave(x_batch, I_cond)

            # Distillation loss: minimise MSE between predicted and target distributions
            loss = nn.functional.mse_loss(mu_pred, mu_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / max(len(data_loader), 1)
        print(f"    [Distiller] Epoch {epoch + 1}/{num_epochs}, Loss: {avg:.6f}")

    # --- Compute stable Master-Tone by averaging over all training batches ---
    producer.eval()
    all_tones: list[torch.Tensor] = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_flat = x_batch.view(x_batch.size(0), -1).to(device)
            tone = producer(x_flat)       # (master_tone_dim,)
            all_tones.append(tone)

    I_new = torch.stack(all_tones).mean(dim=0).detach()
    return I_new
