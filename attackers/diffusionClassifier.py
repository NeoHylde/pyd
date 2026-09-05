# Implements the Algorithm 1 Diffusion Classifier from:
# Li, A. C., Prabhudesai, M., Duggal, S., Brown, E., & Pathak, D. (2023, September 13). 
# Your diffusion model is secretly a zero-shot classifier. 
# arXiv.org. https://arxiv.org/abs/2303.16203 

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
scheduler = pipe.scheduler
unet = pipe.unet

@torch.no_grad()
def encode_prompts(class_prompts):
    tokens = tokenizer(class_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(text_encoder.device)

    c = text_encoder(**tokens).last_hidden_state
    return c

@torch.no_grad()
def diffusion_classifier(image_path, class_prompts, trials):
    cond_inp = encode_prompts(class_prompts)
    
    # Prep input
    x = Image.open(image_path).convert("RGB")
    
    import torchvision.transforms.functional as TF
    x = x.resize((512, 512))
    img_tensor = TF.to_tensor(x).unsqueeze(0) * 2.0 - 1.0
    img_tensor = img_tensor.to("cuda", torch.float16)
    
    # Compress image to latent
    posterior = vae.encode(img_tensor).latent_dist
    latents = posterior.sample() * 0.18215
    
    # List of loss error's for prompt c.
    errors = [[] for _ in range(len(cond_inp))]
    
    # Monte Carlo loop
    for i in range(trials): 
        # Sample random timestamp t
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=latents.device)
        # Sample gaussian noise ε
        noise = torch.randn_like(latents)
        
        # x_t = sqrt(ᾱ_t) * latents + sqrt(1 - ᾱ_t) * noise
        # add noise to image at t
        x_t = scheduler.add_noise(latents, noise, t)
        # Loop over each encoded prompt
        for k in range(len(cond_inp)):
            c_k = cond_inp[k:k+1]
            # UNET outputs expected noise at t
            pred = unet(x_t, t, encoder_hidden_states=c_k).sample
            # Noise-pred error for (t, ε) under prompt c_k
            err = ((noise - pred) ** 2).mean()
            errors[k].append(err.item())

    # Take smallest mean loss for each encoded prompt, this has highest prob of being the "correct" prompt
    mean_errors = [np.mean(errors[i]) for i in range(len(cond_inp))]
    best = np.argmin(mean_errors)
    
    return class_prompts[best]

if __name__ == "__main__":
    class_prompts = [
        "a photo of a Labrador retriever",
        "a photo of a golden retriever",
        "a photo of a tabby cat",
        "a photo of a school bus",
        "a photo of an iguana",
    ]
    result = diffusion_classifier("fgsm_img.jpg", class_prompts, trials=40)
    base = diffusion_classifier("YellowLabradorLooking_new.jpg", class_prompts, trials=40)
    print(base)
    print(result)
    