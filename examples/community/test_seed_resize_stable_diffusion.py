import torch as th
import numpy as np
from diffusers import DiffusionPipeline

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

pipe = DiffusionPipeline.from_pretrained(
    "/Users/zhoubingzheng/projects/huggingface/stable-diffusion-v1-5",
    use_auth_token=True,
    custom_pipeline="seed_resize_stable_diffusion"
).to(device)

def dummy(images, **kwargs):
    return images, False

pipe.safety_checker = dummy


images = []
th.manual_seed(0)
generator = th.Generator("cpu").manual_seed(0)

seed = 0
prompt = "A painting of a futuristic cop"

width = 512
height = 512

res = pipe(
    prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=height,
    width=width,
    generator=generator)
image = res.images[0]
image.save('./examples/community/seed_resize/seed_resize_{w}_{h}_image.png'.format(w=width, h=height))


th.manual_seed(0)
generator = th.Generator("cpu").manual_seed(0)

pipe = DiffusionPipeline.from_pretrained(
    "/Users/zhoubingzheng/projects/huggingface/stable-diffusion-v1-5",
    custom_pipeline="/Users/zhoubingzheng/projects/diffusers/examples/community/seed_resize_stable_diffusion.py"
).to(device)

width = 512
height = 592

res = pipe(
    prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=height,
    width=width,
    generator=generator)
image = res.images[0]
image.save('./examples/community/seed_resize/seed_resize_{w}_{h}_image.png'.format(w=width, h=height))

pipe_compare = DiffusionPipeline.from_pretrained(
    "/Users/zhoubingzheng/projects/huggingface/stable-diffusion-v1-5",
    custom_pipeline="/Users/zhoubingzheng/projects/diffusers/examples/community/seed_resize_stable_diffusion.py"
).to(device)

res = pipe_compare(
    prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=height,
    width=width,
    generator=generator
)

image = res.images[0]
image.save('./examples/community/seed_resize/seed_resize_{w}_{h}_image_compare.png'.format(w=width, h=height))