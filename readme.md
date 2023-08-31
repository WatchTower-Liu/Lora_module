# 一个简单的Lora轮子

这是一个简单的Lora轮子，目标是使用最少的代码，实现一个可以进行训练，持久化的微型Lora训练框架
<font color='yellow'> 注意：结果并不兼容webUI以及其他框架 </font>

## 环境说明

代码的目标是尽可能的关注Lora的实现，以最少的代码实现一个可用的Lora，所以整个代码是以**huggingface**的**diffusers**库进行开发的，所需要的环境依赖见<font color='NavyBlue'> requirements.txt </font>

<font color='yellow'> 不支持SDXL </font>

## 运行

安装运行环境

```shell
pip install -r requirements.txt
```

训练以及推理的代码主要在trainLora.py中

```shell
python trainLora.py
```

数据放置到自定义路径，数据使用文本图像对的形式（sd-scripts的数据可以直接使用）

```python
# 设置哪些module参与Lora注入
target_module = ["to_q", "to_k", "to_v", "conv1", "conv2", "proj_in", "proj_out", "conv", "conv_out", "proj", "ff.net.2"] # 选择注入的模块

# 选择基础模型，目前支持diffusers模型与ldm模型（webui中的checkpoint）
# model_name = "runwayml/stable-diffusion-v1-5"
model_name = "F:/local_model/dreamshaper_331BakedVae.safetensors"

# 实例化一个训练器
trainer = train_lora(model_name,     # 模型名称
                   True,             # 是否进行Lora输入
                   target_module=target_module,   # 如果注入，需要修改的module
                   is_diffusers=False,       # 是否是diffusers模型
                   only_local_files=False)   # 是否只使用diffusers模型

trainer.train(40, "./data/your_data")    # 训练
trainer.loraIn.save_lora("./save/lora.pt")  # 持久化

generate_image(4, trainer, "prompt")      # 生成图像

```

## 效果

运行效果基于<font color='NavyBlue'> runwayml/stable-diffusion-v1-5 与 非diffusers模型 </font>。
<font color="yellow">代码重点是为了用最少的编码实现可用的训练，用于明晰原理，不过分追求效果</font>

|diffusers|prompt|  角色/类型  |  原模型   |  Lora   |
|--------|---------|----------|-----------|-----------|
|否|1girl, circle_cut, clothing_cutout, cover, letterboxed, medium_breasts, pillarboxed|KOSMOS|![](./asset/1/0_org.png)|![](./asset/1/0.png)
|是|small_scene, 3D_scene, grass, tree_stump, picnic, faux_figurine|scene|![](./asset/2/3_org.png)|![](./asset/2/3.png)|
|否|small_scene, 3D_scene, sunset, mountain, cloud, gradient_sky, twilight, orange_sky, evening, red_sky|scene|![](./asset/3/3_org.png)|![](./asset/3/3.png)

