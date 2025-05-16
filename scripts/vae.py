from diffusers import AutoencoderKL
import torch

class VAE:
    def __init__(self, device='cuda'):
        self.device = device
        # Use full absolute path and force local files only
        self.model_path = '/content/MuseTalk/models/sd-vae-ft-mse'
        self.vae = AutoencoderKL.from_pretrained(
            self.model_path,
            local_files_only=True  # Only use local files, never try HuggingFace
        )
        self.vae.to(self.device)

    def get_latents_for_unet(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1.0  # normalization
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype('uint8')
        return image