from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
import numpy as np
import time
from safetensors.torch import load_file

from lora import loraModle

from modelDataset import DreamBoothDataset
from utils import convert_ldm_unet_checkpoint

torch.cuda.is_available()

class train_lora():
    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
    cache_dir = "F:/huggingface_model/"    # diffusers的本地缓存路径
    def __init__(self, 
                 base_model_name:str,     
                 injectLora:bool,         
                 multiplier:float = 1.0,
                 is_diffusers:bool = True,
                 target_module:list = None, 
                 lr:float = 1e-4,
                 sampleTimeStep:int = 30, 
                 lora_dim: int= 32, 
                 lora_alpha:int = 32, 
                 train_img_size:int = 512,
                 only_local_files:bool = True):
        self.base_model_name = base_model_name
        self.only_local_files = only_local_files
        self.sampleTimeStep = sampleTimeStep
        self.is_diffusers = is_diffusers
        self.load_model()
        self.getLatent_model()
        self.load_text_encoder()
        self.load_scheduler()
        
        self.lr = lr
        
        self.train_img_size = train_img_size
        self.allTimestep = self.ddim.timesteps

        self.image_processor = VaeImageProcessor()
        if injectLora:
            self.loraIn = loraModle(target_module, lora_dim, lora_alpha, multiplier)
            lora_unet = self.loraIn.inject(self.unet)
            self.unet = lora_unet
        
    def train(self, epoch:int, image_path:str):
        dataset = DreamBoothDataset(image_path, self.tokenizer, size=self.train_img_size)
        optim = torch.optim.AdamW(self.loraIn.lora_parameter, lr=self.lr)
        loss_fn = nn.MSELoss()
        dataL = DataLoader(dataset, batch_size=1, shuffle=True)
        # self.unet.requires_grad_(False)
        self.unet.eval()
        for e in range(epoch):
            eloss = 0
            for example in tqdm(dataL):
                text = example["instance_prompt_ids"][:,0,:]
                train_data = example["instance_images"].cuda()
                emb = self.encode_training_prompt(text).detach().requires_grad_(False)
                latent = self.getImgLatent(train_data).detach().requires_grad_(False)
       
                Ti = torch.randint(0, self.ddim.config.num_train_timesteps, (1,)).long().cuda()
                # Ti = self.ddim.timesteps[Tin]
                optim.zero_grad()
                noise = torch.randn_like(latent).cuda()
            
                latent_model_input = self.addNoise(latent, noise, Ti)
                latent_model_input = self.ddim.scale_model_input(latent_model_input, Ti)
                noise_pred = self.unet(latent_model_input, Ti, encoder_hidden_states = emb).sample
                if self.ddim.config.prediction_type == "epsilon":
                    target = noise
                elif self.ddim.config.prediction_type == "v_prediction":
                    target = self.ddim.get_velocity(latent, noise, Ti)
                loss = loss_fn(target, noise_pred)

                loss.backward()
                optim.step()
                eloss += loss.detach().cpu().numpy()
            eloss /= len(self.allTimestep)

            print("epoch:{} loss:{}".format(e, eloss))

    def addNoise(self, latent:torch.Tensor, noise: torch.Tensor, timestep:torch.Tensor):
        return self.ddim.add_noise(latent, noise, timestep)
    
    def load_scheduler(self):
        if not self.is_diffusers:
            MN = self.DEFAULT_MODEL
        else:
            MN = self.base_model_name
        self.ddim = DDIMScheduler.from_pretrained(MN, 
                                                  subfolder="scheduler", 
                                                  local_files_only=self.only_local_files, 
                                                #   torch_dtype=torch.float16, 
                                                  use_safetensors=True, 
                                                  cache_dir = self.cache_dir)
        

        self.ddim.set_timesteps(self.sampleTimeStep, device="cuda:0")
        
    def sample_step(self, latent: torch.Tensor, niose: torch.Tensor, timestep: torch.Tensor):
        return self.ddim.step(niose, timestep, latent)['prev_sample']

    
    def make_noise(self, latent_size: list):
        latent = torch.randn(1, *latent_size).cuda()
        latent = latent * self.ddim.init_noise_sigma
        return  latent

    def sample(self, latent:torch.Tensor, prompt_embeds:torch.Tensor, guidance_scale:int=7):
        # print(prompt_embeds.shape)
        # print(latent.shape)
        for Tin in tqdm(range(len(self.allTimestep))):
            Ti = self.allTimestep[Tin]
            latent_model_input = torch.cat([latent] * 2)
            # print(latent)
            latent_model_input = self.ddim.scale_model_input(latent_model_input, Ti)
            noise_pred = self.unet(latent_model_input, Ti, encoder_hidden_states = prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # print(noise)
            latent = self.sample_step(latent, noise_pred, Ti)
        return latent

    def load_model(self):
        if self.is_diffusers:
            self.unet = UNet2DConditionModel.from_pretrained(self.base_model_name, 
                                                             local_files_only = self.only_local_files, 
                                                            #  torch_dtype=torch.float16, 
                                                             use_safetensors=True, 
                                                             subfolder = "unet",
                                                             cache_dir = self.cache_dir).cuda()
        else:
            state_dict = load_file(self.base_model_name)  # 加载ldm参数

            unet_config = UNet2DConditionModel.load_config(self.DEFAULT_MODEL, 
                                                local_files_only = self.only_local_files, 
                                            #  torch_dtype=torch.float16, 
                                                use_safetensors=True, 
                                                subfolder = "unet",
                                                cache_dir = self.cache_dir)
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)  # 转换ldm参数到diffusers

            self.unet = UNet2DConditionModel(**unet_config).to("cuda:0")
            self.unet.load_state_dict(converted_unet_checkpoint)

        
        self.unet.enable_xformers_memory_efficient_attention()
        
    def getLatent_model(self):
        if not self.is_diffusers:
            MN = self.DEFAULT_MODEL
        else:
            MN = self.base_model_name
        self.vae = AutoencoderKL.from_pretrained(MN, 
                                                 local_files_only = self.only_local_files,
                                                #  torch_dtype=torch.float16,
                                                #  use_safetensors=True,
                                                 subfolder = "vae",
                                                 cache_dir = self.cache_dir).cuda()
        

    def load_text_encoder(self):
        if not self.is_diffusers:
            MN = self.DEFAULT_MODEL
        else:
            MN = self.base_model_name
        self.text_encoder = CLIPTextModel.from_pretrained(MN, 
                                                          local_files_only = self.only_local_files,
                                                        #   torch_dtype=torch.float16,
                                                        #   use_safetensors=True,
                                                          subfolder = "text_encoder",
                                                          cache_dir = self.cache_dir).cuda()
    
        self.tokenizer = CLIPTokenizer.from_pretrained(MN,
                                                         local_files_only = self.only_local_files,
                                                         subfolder = "tokenizer",
                                                         cache_dir = self.cache_dir)
        
        
    @staticmethod
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids 

    def encode_training_prompt(self, text_input_ids):
        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds
    
    def encode_prompt(self, prompt:str, neg_prompt:str = None):
        text_input_ids = self.tokenize_prompt(self.tokenizer, prompt)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )

        prompt_embeds = prompt_embeds[0]

        if neg_prompt is None:
            neg_prompt = ""
        negative_text_input_ids = self.tokenize_prompt(self.tokenizer, neg_prompt)
        negative_prompt_embeds = self.text_encoder(
            negative_text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
    
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def get_text_embedding(self, prompt:str, neg_prompt:str = None):
        return self.encode_prompt(prompt, neg_prompt)

        
    def getImgLatent(self, img:torch.Tensor):
        # img = self.image_processor.preprocess(img)
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
    
    def getImg(self, latent):
        image = self.vae.decode(latent / self.vae.config.scaling_factor)[0]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type="np", do_denormalize=[True])
        return image

def generate_image(generate_num: int, 
                   trainer:train_lora, 
                   prompt:str, 
                   image_size:int = 512,
                   guide_image: np.ndarray = None, 
                   neg_prompt:str = None, 
                   guidance_scale:int=7, 
                   save_path:str = "save"):
    if guide_image is not None:
        torchImg = torch.from_numpy(guide_image).permute(2, 0, 1).unsqueeze(0).cuda().float()
        # torchImg = (torchImg - torchImg.min()) / (torchImg.max() - torchImg.min())
        torchImg = torchImg / 255
        latent = trainer.getImgLatent(torchImg)
        
    for i in range(generate_num):
        if guide_image is not None:
            latent = trainer.addNoise(latent, torch.randn_like(latent), trainer.ddim.timesteps[0])   # img to img
        else:
            latent = trainer.make_noise([4, image_size//8, image_size//8])
        prompt_embeds = trainer.get_text_embedding(prompt, neg_prompt)
        # print(prompt_embedding)
        with torch.no_grad():
            latent = trainer.sample(latent, prompt_embeds, guidance_scale)
        img = trainer.getImg(latent.float())
        print(img.shape)
        img = img[0]
        # img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype("uint8")
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite("{}/{}.png".format(save_path, i), img)
    
def main():
    target_module = ["to_q", "to_k", "to_v", "conv1", "conv2", "proj_in", "proj_out", "conv", "conv_out", "proj", "ff.net.2"] # 选择注入的模块

    # model_name = "runwayml/stable-diffusion-v1-5"
    model_name = "F:/local_model/dreamshaper_331BakedVae.safetensors"

    trainer = train_lora(model_name, True, 
                         target_module=target_module, 
                         is_diffusers=False,
                         only_local_files=False)
    
    # print(trainer.unet)     # 打印模型查看哪些需要注入
    # exit()


    trainer.train(40, "./data/your_data")    # 训练
    trainer.loraIn.save_lora("./save/lora.pt")
    
    new_trainer = train_lora(model_name, True,   
                             multiplier = 0.8,      
                             target_module=target_module,
                             is_diffusers=False,
                             only_local_files=False)               # 在新的模型上注入，测试保存的模型是否可用
    new_trainer.loraIn.load_lora("./save/lora.pt")

    # img = cv.imread("./data/xiang/5d839a7bf712838d3ec07112d699037f.jpeg")
    # img = cv.resize(img, (768, 768))

    generate_image(4, new_trainer, "prompt")


if __name__ == "__main__":
    main()

