# 4bit_quantization-Nunchaku-for-Qwen-imageS-supporting-LoRA
最新的图像生成4bit量化工程Nunchaku在图像生成模型取得了突破性的进展，但目前官方的推理代码不支持Qwen-image系的LoRA加载。
因此对官方推理代码进行了修改，对accelerate框架的LoRA进行了适配，使用时仅需导入satetensor文件的绝对路径（不能出现中文、空格）。
docker：registry.intsig.net/zihan_huang/svdquant_lora

快速开始：
1.下载NUNCHAKU官方演示项目： https://github.com/nunchaku-tech/nunchaku

2.将 /nunchaku-main/examples/v1/qwen-image.py 替换为项目文件 qwen-image.py

3.cd /nunchaku-main/examples/v1 将项目文件 lora.py 导入v1文件夹

4.启动docker环境： docker run -it -v /juicefs-algorithm/:/juicefs-algorithm/ -v /DeepLearning/:/DeepLearning/ --gpus all --shm-size=20G registry.intsig.net/zihan_huang/svdquant_lora

5.激活conda环境 conda activate nunchaku

6.设置好各项参数路径（模型参数、lora参数、output）

7.python /nunchaku-main/examples/v1/qwen-image.py
