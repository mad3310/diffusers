from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_path = "/Users/zhoubingzheng/projects/huggingface/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("mps")

pipe.unet.load_attn_procs("/Users/zhoubingzheng/projects/huggingface/dreambooth-lora-model")

prompt = "a photo of handsome chinese guy in suit walking on the tall buildings street, big eyes, a BMW car stop on the street, tall buildings, \
    summary, highly detailed, 4k"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("./examples/dreambooth/images/dreambooth_lora.png")


#crowds in the street, celebrate china new year, spring festival, digital painting, highly detailed, masterpiece, concept art
#cliff, travel, highly detailed, masterpiece, beautiful painting, 4k
#a portrait of a girl surrounded by delicate feathers, face, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by Krenz Cushart and Artem Demura and alphonse mucha
#portrait of Gypsy woman, big eyes, highly detailed, masterpiece, realistic, light

#Athletes in stadium, soccer match, multiplayer, world cup, digital painting, highly detailed, masterpiece, concept art
#A private party, beautiful men and women, everyone drinking and talking, highly detailed, masterpiece, light, smooth, illustration, sharp focus, 4k
#A handsome fashion guy in China, wearing a blue striped suit stand on the street, wears famous watches on hand, wearing splendid collar, the background is a fashionable neighborhood
#a portrait of chinese fashion guy smoking and walking on the street, the backgroud exist on BMW, highly detailed, realistic, light, 4k
#A high-speed train runs in the desert, and there is a forest deep in the desert, highly detailed, realistic, 4k
#A Ukrainian beauty girl is lying on the green large lawn, a castle in the distance, highly detailed, big eyes, 4k
#China and the U.S. Navy fired mutual missiles in the Pacific Ocean, highly detailed, 4k
#Optimus Prime and Megatron are fighting in space, highly detailed, 4k
#Spaceship travels in space
#an Aircraft carrier, 4 frigates, 2 submarines, highly detailed, 4k
#bamboo forest over the sea, highly detailed, 4k
#The Color Of The Night on the sun, crowds in the street, highly detailed, 4k
#The Color Of the city Night, highly detailed, 4k
#a photo of handsome chinese guy in suit walking on the street, BMW, tall buildings, summary, highly detailed, 4k
