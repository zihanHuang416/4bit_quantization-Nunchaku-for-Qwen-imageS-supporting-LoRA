import torch
from diffusers import QwenImagePipeline
import os # 导入 os 模块
import time # 导入 time 模块
import sys
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from lora import load_lora_and_print_keys

rank = 128
target_device = "cuda:0"
target_lora_path = "/.safetensors" #lora_path

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"/nunchaku-tech/nunchaku-qwen-image/svdq-int4_r128-qwen-image.safetensors", #与rank对应
    device=target_device
)
# 加载lora
lora = load_lora_and_print_keys(target_lora_path)
#lora强度
lora_strength = 1
with torch.no_grad():
    for i, block in enumerate(transformer.transformer_blocks):
        #to_out0
        block.attn.to_out[0].proj_down_lora = lora[f"transformer_blocks.{i}.attn.to_out.0.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.to_out[0].proj_up_lora = lora[f"transformer_blocks.{i}.attn.to_out.0.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #to_add_out
        block.attn.to_add_out.proj_down_lora = lora[f"transformer_blocks.{i}.attn.to_add_out.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.to_add_out.proj_up_lora = lora[f"transformer_blocks.{i}.attn.to_add_out.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #img_mod1
        block.img_mod[1].proj_down_lora = lora[f"transformer_blocks.{i}.img_mod.1.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.img_mod[1].proj_up_lora = lora[f"transformer_blocks.{i}.img_mod.1.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #txt_mod1
        block.txt_mod[1].proj_down_lora = lora[f"transformer_blocks.{i}.txt_mod.1.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.txt_mod[1].proj_up_lora = lora[f"transformer_blocks.{i}.txt_mod.1.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #to_qkv
        block.attn.to_qkv.q_dowm = lora[f"transformer_blocks.{i}.attn.to_q.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.to_qkv.q_up = lora[f"transformer_blocks.{i}.attn.to_q.lora_B.default.weight"].mul(lora_strength).to(target_device)
        block.attn.to_qkv.k_dowm = lora[f"transformer_blocks.{i}.attn.to_k.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.to_qkv.k_up = lora[f"transformer_blocks.{i}.attn.to_k.lora_B.default.weight"].mul(lora_strength).to(target_device)
        block.attn.to_qkv.v_dowm = lora[f"transformer_blocks.{i}.attn.to_v.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.to_qkv.v_up = lora[f"transformer_blocks.{i}.attn.to_v.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #add_qkv
        block.attn.add_qkv_proj.q_dowm = lora[f"transformer_blocks.{i}.attn.add_q_proj.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.add_qkv_proj.q_up = lora[f"transformer_blocks.{i}.attn.add_q_proj.lora_B.default.weight"].mul(lora_strength).to(target_device)
        block.attn.add_qkv_proj.k_dowm = lora[f"transformer_blocks.{i}.attn.add_k_proj.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.add_qkv_proj.k_up = lora[f"transformer_blocks.{i}.attn.add_k_proj.lora_B.default.weight"].mul(lora_strength).to(target_device)
        block.attn.add_qkv_proj.v_dowm = lora[f"transformer_blocks.{i}.attn.add_v_proj.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.attn.add_qkv_proj.v_up = lora[f"transformer_blocks.{i}.attn.add_v_proj.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #img_mlp2
        block.img_mlp.net[2].proj_down_lora = lora[f"transformer_blocks.{i}.img_mlp.net.2.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.img_mlp.net[2].proj_up_lora = lora[f"transformer_blocks.{i}.img_mlp.net.2.lora_B.default.weight"].mul(lora_strength).to(target_device)
        #txt_mlp2
        block.txt_mlp.net[2].proj_down_lora = lora[f"transformer_blocks.{i}.txt_mlp.net.2.lora_A.default.weight"].T.mul(lora_strength).to(target_device)
        block.txt_mlp.net[2].proj_up_lora = lora[f"transformer_blocks.{i}.txt_mlp.net.2.lora_B.default.weight"].mul(lora_strength).to(target_device)

transformer.set_offload(
    True, use_pin_memory=False, num_blocks_on_gpu=60, use_cpu=False, offload_gpu_device=target_device
)

pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", #model_path
    transformer=transformer,#这里选择是否使用量化后的模块
    torch_dtype=torch.bfloat16
)
pipe.to(target_device)

prompt = """帮我做一张海报，画面中文字内容是：                                                                 主标题："立夏轻甜上新 一杯清爽迎夏来" 内容： "立夏当日，到店点单，立夏限定新品尝鲜。 消费参与，新品打卡分享赠小料券，新品与经典款同步供应。" 补充信息： "活动时间：立夏当日（2025 年 5 月 6 日） 活动主题：以立夏节气为灵感，推出清爽口感奶茶新品 产品亮点：选用当季鲜果 / 草本原料，主打清爽不腻口感，适配初夏味蕾 画面元素:"中心是一杯巨大的新品奶茶在一个巨大的荔枝微缩世界场景（有工人，有荔枝树、有草地，有阳光），瓶子上贴着荔枝的图案的doodle贴纸，奶盖泡沫里浮出三～四颗白色荔枝果肉，特写镜头，微距摄影" """
negative_prompt = " "

for prompt_test in all_prompts:
    #平均测试
    test_runs = 1
    for i in range(test_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize() # 确保所有之前的CUDA操作完成
        start_time = time.perf_counter() # 使用 perf_counter 提供高精度计时

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=1080,
            height=1920,
            num_inference_steps=50,
            true_cfg_scale=4.0,
        ).images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize() # 确保所有GPU操作完成
        
        end_time = time.perf_counter() # 结束计时
        latency = end_time - start_time
        print(f"测试运行 {i+1}/{test_runs} 延迟: {latency:.2f} 秒")

    # 保存图片
    output_dir = "/nunchaku-main/outputs" #output_path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"qwen-image-r{rank}-lora-19000.png")
    image.save(output_path)
    print(f"✅ 图片已成功保存到: {output_path}")
