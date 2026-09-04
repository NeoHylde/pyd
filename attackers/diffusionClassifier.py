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
    
    x = Image.open(image_path).convert("RGB")
     
    import torchvision.transforms.functional as TF
    x = x.resize((512, 512))
    img_tensor = TF.to_tensor(x).unsqueeze(0) * 2.0 - 1.0
    img_tensor = img_tensor.to("cuda", torch.float16)
    
    
    posterior = vae.encode(img_tensor).latent_dist
    latents = posterior.sample() * 0.18215
    
    errors = [[] for _ in range(len(cond_inp))]
    for i in range(trials):
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=latents.device)
        noise = torch.randn_like(latents)
        
        x_t = scheduler.add_noise(latents, noise, t)
        for k in range(len(cond_inp)):
            c_k = cond_inp[k:k+1]
            pred = unet(x_t, t, encoder_hidden_states=c_k).sample
            err = ((noise - pred) ** 2).mean()
            errors[k].append(err.item())

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
    result = diffusion_classifier("fgsm_img.jpg", class_prompts, trials=20)
    base = diffusion_classifier("YellowLabradorLooking_new.jpg", class_prompts, trials=20)
    print(base)
    print(result)
    