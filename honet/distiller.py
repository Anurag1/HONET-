import torch
import torch.nn as nn

def distill_knowledge_to_master_tone(octave, data_loader, prev_tone, device, master_tone_dim, z_dim):
    """Simulates the functional distillation process for demonstration."""
    print("  Distilling knowledge into a new Master-Tone...")
    octave.eval()
    all_latents = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            I_cond_batch = prev_tone.unsqueeze(0).expand(x_batch.size(0), -1) if prev_tone is not None else None
            _, mu, _ = octave(x_batch, I_cond_batch)
            all_latents.append(mu)
    
    avg_latent = torch.cat(all_latents).mean(dim=0)
    # Simulate a small learned projection layer
    projection = nn.Linear(z_dim, master_tone_dim).to(device)
    master_tone = projection(avg_latent).detach()
    
    print(f"  Distillation complete. Locked Master-Tone with shape: {master_tone.shape}")
    return master_tone