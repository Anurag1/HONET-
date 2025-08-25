import torch
import torch.nn as nn

class AbstractOctave(nn.Module):
    """Abstract Base Class for all Octave modules."""
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x, I=None): raise NotImplementedError

class ImageOctave(AbstractOctave):
    def __init__(self, z_dim, c_dim, channels, size):
        super().__init__()
        self.size = size
        self.g_net_conv = nn.Sequential(nn.Conv2d(channels, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.g_net_fc = nn.Linear(64 * (size//4)**2 + c_dim, z_dim * 2)
        self.r_net_fc = nn.Linear(z_dim + c_dim, 64 * (size//4)**2)
        self.r_net_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, channels, 4, 2, 1), nn.Sigmoid())
    def forward(self, x, I=None):
        if I is None: I = torch.zeros(x.shape[0], self.g_net_fc.in_features - 64 * (self.size//4)**2).to(x.device)
        h = self.g_net_conv(x).view(x.size(0), -1); h_cond = torch.cat([h, I], dim=1)
        mu, logvar = self.g_net_fc(h_cond).chunk(2, dim=1); z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, I], dim=1)
        h_recon = self.r_net_fc(z_cond).view(z.size(0), 64, self.size//4, self.size//4)
        return self.r_net_conv(h_recon), mu, logvar

class TabularOctave(AbstractOctave):
    def __init__(self, z_dim, c_dim, in_dim):
        super().__init__()
        self.g_net = nn.Sequential(nn.Linear(in_dim + c_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.g_fc = nn.Linear(128, z_dim * 2)
        self.r_net = nn.Sequential(nn.Linear(z_dim + c_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, in_dim))
    def forward(self, x, I=None):
        if I is None: I = torch.zeros(x.shape[0], self.g_net[0].in_features - x.shape[1]).to(x.device)
        h = self.g_net(torch.cat([x, I], dim=1)); mu, logvar = self.g_fc(h).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.r_net(torch.cat([z, I], dim=1)), mu, logvar

class SequentialOctave(AbstractOctave):
    def __init__(self, z_dim, c_dim, in_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.g_net_lstm = nn.LSTM(in_dim, 128, batch_first=True)
        self.g_fc = nn.Linear(128 + c_dim, z_dim * 2)
        self.r_fc1 = nn.Linear(z_dim + c_dim, 128)
        self.r_net_lstm = nn.LSTM(in_dim, 128, batch_first=True)
        self.r_fc2 = nn.Linear(128, in_dim)
    def forward(self, x, I=None):
        if I is None: I = torch.zeros(x.shape[0], self.g_fc.in_features - 128).to(x.device)
        _, (h, _) = self.g_net_lstm(x); mu, logvar = self.g_fc(torch.cat([h.squeeze(0), I], dim=1)).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar); h0 = self.r_fc1(torch.cat([z, I], dim=1)).unsqueeze(0); c0 = torch.zeros_like(h0)
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(x.device); outputs = []
        for _ in range(self.seq_len):
            output, (h0, c0) = self.r_net_lstm(decoder_input, (h0, c0))
            output = self.r_fc2(output); outputs.append(output); decoder_input = output
        return torch.cat(outputs, dim=1), mu, logvar