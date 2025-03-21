# W/O Camera pipeline

- **demo_gpt_images.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>` ⇒ **GPT-4o** `<Situation Understanding>`
    - image directory -> result directory

# W/ Camera pipeline
- **demo.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>`

- **demo_fl2+gpt.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>` ⇒ **GPT-4o** `<Situation Understanding>`

- **demo_only_gpt.py**
    - **GPT-4o** `<Situation Understanding>`, `<OV-OD>` 


# Prerequisite 
```bash 
Python version: 3.11.9
OpenCV version: 4.10.0
Pillow (PIL) version: 10.4.0
PyTorch version: 2.4.0
    torch==2.4.0
    torchaudio==2.4.0
    torchelastic==0.2.2
    torchvision==0.19.0
CUDA version: 12.1
Transformers version: 4.44.0
    einops==0.8.0
    flash-attn==2.6.3
    timm==1.0.8
```
```bash
pip install opencv-python timm einops flash_attn transformers
```

### Openai API Key 입력
```bash
export OPENAI_API_KEY=<sk-proj-...>
```

ImportError 발생 시
``` bash
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get install -y libgl1-mesa-glx libglib2.0-0
# pip install opencv-python timm einops flash_attn transformers
```

