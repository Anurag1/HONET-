"""
honet/octaves.py
----------------
Defines the core Octave modules for the HONet architecture.

Each Octave is a Conditional Variational Autoencoder (CVAE) that:
  1. Encodes input data into a latent distribution (mu, logvar) conditioned
     on an optional Master-Tone vector I.
  2. Decodes a sampled latent vector z (also conditioned on I) back to the
     original input space.

Three modality-specific Octaves are provided:
  - ImageOctave     : for image data (CNN-based encoder/decoder)
  - TabularOctave   : for tabular / flat feature data (MLP-based)
  - SequentialOctave: for time-series / sequential data (LSTM-based)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared helper: reparameterization trick
# ---------------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample z ~ N(mu, sigma^2) using the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# ImageOctave  (CNN-based CVAE for image data)
# ---------------------------------------------------------------------------

class ImageOctave(nn.Module):
    """
    Conditional VAE for image data.

    Architecture
    ------------
    Encoder (G-Net):
        Conv2d stack -> flatten -> FC -> (mu, logvar)  [conditioned on I]

    Decoder (F-Net):
        FC -> reshape -> ConvTranspose2d stack -> reconstructed image [conditioned on I]

    Parameters
    ----------
    z_dim          : dimensionality of the latent space
    master_tone_dim: dimensionality of the conditioning Master-Tone vector
    img_channels   : number of image channels (1 for MNIST, 3 for CIFAR-10)
    img_size       : spatial resolution (assumed square, e.g. 28 or 32)
    """

    def __init__(
        self,
        z_dim: int = 32,
        master_tone_dim: int = 64,
        img_channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.master_tone_dim = master_tone_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # ---------- Encoder ----------
        self.g_net_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self._conv_out_size = img_size // 4  # after two stride-2 convolutions
        conv_flat_dim = 64 * self._conv_out_size * self._conv_out_size

        # Conditioned FC: conv features + I -> mu & logvar
        self.g_net_fc = nn.Linear(conv_flat_dim + master_tone_dim, z_dim * 2)

        # ---------- Decoder ----------
        # Conditioned FC: z + I -> feature map
        self.f_net_fc = nn.Linear(z_dim + master_tone_dim, conv_flat_dim)

        self.f_net_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, I: torch.Tensor | None = None) -> tuple:
        h = self.g_net_conv(x).view(x.size(0), -1)
        if I is not None:
            h = torch.cat([h, I], dim=1)
        else:
            # Pad with zeros when no conditioning is provided (first task)
            pad = torch.zeros(x.size(0), self.master_tone_dim, device=x.device)
            h = torch.cat([h, pad], dim=1)
        out = self.g_net_fc(h)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor, I: torch.Tensor | None = None) -> torch.Tensor:
        if I is not None:
            h = torch.cat([z, I], dim=1)
        else:
            pad = torch.zeros(z.size(0), self.master_tone_dim, device=z.device)
            h = torch.cat([z, pad], dim=1)
        h = self.f_net_fc(h)
        h = h.view(h.size(0), 64, self._conv_out_size, self._conv_out_size)
        return self.f_net_deconv(h)

    def forward(
        self,
        x: torch.Tensor,
        I: torch.Tensor | None = None,
    ) -> tuple:
        mu, logvar = self.encode(x, I)
        z = reparameterize(mu, logvar)
        x_recon = self.decode(z, I)
        return x_recon, mu, logvar


# ---------------------------------------------------------------------------
# TabularOctave  (MLP-based CVAE for tabular/flat data)
# ---------------------------------------------------------------------------

class TabularOctave(nn.Module):
    """
    Conditional VAE for tabular (flat feature) data.

    Parameters
    ----------
    z_dim          : dimensionality of the latent space
    master_tone_dim: dimensionality of the conditioning Master-Tone vector
    input_dim      : number of input features
    hidden_dim     : hidden layer width for the MLP blocks
    """

    def __init__(
        self,
        z_dim: int = 32,
        master_tone_dim: int = 64,
        input_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.master_tone_dim = master_tone_dim
        self.input_dim = input_dim

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + master_tone_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + master_tone_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _pad_condition(self, x: torch.Tensor, I: torch.Tensor | None) -> torch.Tensor:
        if I is not None:
            return torch.cat([x, I], dim=1)
        pad = torch.zeros(x.size(0), self.master_tone_dim, device=x.device)
        return torch.cat([x, pad], dim=1)

    def encode(self, x: torch.Tensor, I: torch.Tensor | None = None) -> tuple:
        h = self.encoder(self._pad_condition(x, I))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, I: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(self._pad_condition(z, I))

    def forward(
        self,
        x: torch.Tensor,
        I: torch.Tensor | None = None,
    ) -> tuple:
        mu, logvar = self.encode(x, I)
        z = reparameterize(mu, logvar)
        return self.decode(z, I), mu, logvar


# ---------------------------------------------------------------------------
# SequentialOctave  (LSTM-based CVAE for time-series data)
# ---------------------------------------------------------------------------

class SequentialOctave(nn.Module):
    """
    Conditional VAE for sequential / time-series data.

    The encoder uses an LSTM to summarise the input sequence into a latent
    distribution.  The decoder uses a transposed LSTM (implemented as a
    sequence-to-sequence model) to reconstruct the original sequence.

    Parameters
    ----------
    z_dim          : dimensionality of the latent space
    master_tone_dim: dimensionality of the conditioning Master-Tone vector
    input_dim      : number of features per time step
    seq_len        : number of time steps in the sequence
    hidden_dim     : LSTM hidden state width
    """

    def __init__(
        self,
        z_dim: int = 32,
        master_tone_dim: int = 64,
        input_dim: int = 1,
        seq_len: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.master_tone_dim = master_tone_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # ---------- Encoder ----------
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # Map final hidden state + condition -> mu, logvar
        self.fc_mu = nn.Linear(hidden_dim + master_tone_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim + master_tone_dim, z_dim)

        # ---------- Decoder ----------
        # Seed: expand z + I to a per-step hidden state seed
        self.decoder_fc = nn.Linear(z_dim + master_tone_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def _pad_condition(self, x: torch.Tensor, I: torch.Tensor | None) -> torch.Tensor:
        if I is not None:
            return torch.cat([x, I], dim=1)
        pad = torch.zeros(x.size(0), self.master_tone_dim, device=x.device)
        return torch.cat([x, pad], dim=1)

    def encode(self, x: torch.Tensor, I: torch.Tensor | None = None) -> tuple:
        # x shape: (batch, seq_len, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n.squeeze(0)  # (batch, hidden_dim)
        h_cond = self._pad_condition(h_n, I)
        return self.fc_mu(h_cond), self.fc_logvar(h_cond)

    def decode(self, z: torch.Tensor, I: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = z.size(0)
        seed = self.decoder_fc(self._pad_condition(z, I))  # (batch, hidden_dim)
        h0 = seed.unsqueeze(0)  # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)

        # Teacher-forcing with zeros as input (free-running generation)
        dec_input = torch.zeros(batch_size, self.seq_len, self.input_dim, device=z.device)
        out, _ = self.decoder_lstm(dec_input, (h0, c0))  # (batch, seq_len, hidden_dim)
        return self.output_fc(out)  # (batch, seq_len, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        I: torch.Tensor | None = None,
    ) -> tuple:
        mu, logvar = self.encode(x, I)
        z = reparameterize(mu, logvar)
        return self.decode(z, I), mu, logvar
