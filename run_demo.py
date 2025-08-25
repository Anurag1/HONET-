import torch
from torch.optim import Adam
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Import project-specific modules
from honet.data_factory import get_task_data
from honet.octaves import ImageOctave, TabularOctave, SequentialOctave
from honet.distiller import distill_knowledge_to_master_tone

# --- Configuration ---
print("--- Initializing HONet Multi-Modal Demonstration ---")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM, MASTER_TONE_DIM = 32, 64
NUM_EPOCHS, BATCH_SIZE, LR = 8, 128, 1e-3
OUTPUT_DIR = "live_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_SEQUENCE = ['IMAGE_MNIST', 'TABULAR_CLUSTERS', 'SEQUENTIAL_WAVES']

def vae_loss_function(x, r, m, l):
    recon_loss = F.mse_loss(r, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + l - m.pow(2) - l.exp())
    return (recon_loss + kld_loss) / x.size(0)

def main():
    trained_octaves, master_tones, task_metadata = [], [], []

    for i, task_name in enumerate(TASK_SEQUENCE):
        print(f"\n--- TASK {i+1}/{len(TASK_SEQUENCE)}: LEARNING '{task_name}' ---")
        loader, meta = get_task_data(task_name, BATCH_SIZE)
        task_metadata.append(meta)

        if meta['type'] == 'image': model = ImageOctave(Z_DIM, MASTER_TONE_DIM, meta['channels'], meta['size'])
        elif meta['type'] == 'tabular': model = TabularOctave(Z_DIM, MASTER_TONE_DIM, meta['input_dim'])
        elif meta['type'] == 'sequential': model = SequentialOctave(Z_DIM, MASTER_TONE_DIM, meta['input_dim'], meta['seq_len'])
        model.to(DEVICE)

        optimizer = Adam(model.parameters(), lr=LR)
        I_condition = master_tones[-1] if master_tones else None

        for epoch in range(NUM_EPOCHS):
            for x_batch, _ in loader:
                x_batch = x_batch.to(DEVICE)
                I_cond_batch = I_condition.unsqueeze(0).expand(x_batch.size(0), -1) if I_condition is not None else None
                x_recon, mu, logvar = model(x_batch, I_cond_batch)
                loss = vae_loss_function(x_batch, x_recon, mu, logvar)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            print(f"  Train Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")
        
        trained_octaves.append(model)
        I_new_skill = distill_knowledge_to_master_tone(model, loader, I_condition, DEVICE, MASTER_TONE_DIM, Z_DIM)
        master_tones.append(I_new_skill)

    print("\n--- FINAL PHASE: Verifying Knowledge Retention Across All Modalities ---")
    for i, octave in enumerate(trained_octaves):
        octave.eval()
        meta = task_metadata[i]
        I_condition_gen = master_tones[i] # Condition on the tone distilled *from* this octave's task
        with torch.no_grad():
            z_sample = torch.randn(64, Z_DIM).to(DEVICE)
            I_cond_batch = I_condition_gen.unsqueeze(0).expand(z_sample.size(0), -1)
            
            # Create a dummy input to trigger the right decoder path for generation
            if meta['type'] == 'image': dummy = torch.zeros(64, meta['channels'], meta['size'], meta['size']).to(DEVICE)
            elif meta['type'] == 'tabular': dummy = torch.zeros(64, meta['input_dim']).to(DEVICE)
            else: dummy = torch.zeros(64, meta['seq_len'], meta['input_dim']).to(DEVICE)
            
            generated, _, _ = octave(dummy, I_cond_batch)
            save_path = f"{OUTPUT_DIR}/demo_task_{i+1}_{meta['name']}"
            
            if meta['type'] == 'image': save_image(generated.cpu(), save_path + '.png', nrow=8)
            else:
                plt.figure(); plt.title(f"Generated Output for {meta['name']}")
                if meta['type'] == 'tabular': plt.scatter(generated[:, 0].cpu(), generated[:, 1].cpu(), alpha=0.7)
                else:
                    for j in range(5): plt.plot(generated[j, :, 0].cpu().numpy())
                plt.savefig(save_path + '.png'); plt.close()

            print(f"Successfully generated verification for '{meta['name']}'. See: {save_path}.png")

if __name__ == '__main__':
    main()