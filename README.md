# 4bit_quantization-Nunchaku-for-Qwen-imageS-supporting-LoRA
最新的图像生成4bit量化工程Nunchaku在图像生成模型取得了突破性的进展，但目前官方的推理代码不支持Qwen-image系的LoRA加载。
因此对官方推理代码进行了修改，对accelerate框架的LoRA进行了适配，使用时仅需导入satetensor文件的绝对路径（不能出现中文、空格）。
docker：registry.intsig.net/zihan_huang/svdquant_lora

