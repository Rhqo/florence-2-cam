# W/O Camera

- **demo_gpt_images.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>` ⇒ **GPT-4o** `<Situation Understanding>`
    - image directory -> result directory

# W/ Camera
- **demo.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>`

- **demo_fl2+gpt.py**
    - **Florence-2** `<Captioning>` ⇒ `<Visual Grounding>` ⇒ **GPT-4o** `<Situation Understanding>`
    - ```
      # 출력 예시
      모니터가 놓인 상황 (gpt: 0.4293s | total: 0.8064s)
      모니터 두 대가 있는 상황 (gpt: 0.7073s | total: 1.1498s)
      모니터 앞에 앉은 상황 (gpt: 0.6782s | total: 1.1239s)
      두 대의 모니터가 있는 상황 (gpt: 0.9649s | total: 1.3370s)
      모니터와 책상이 있는 상황. (gpt: 0.5328s | total: 0.8983s)
      ...
      ```

- **demo_only_gpt.py**
    - **GPT-4o** `<Situation Understanding>`, `<OV-OD>`
    - ```
      # 출력 예시
      {situation}: 사무실에서 작업 중인 상황  
      {location}: keyboard: [100, 200, 300, 250]  
                  monitor1: [320, 150, 600, 300]  
                  monitor2: [610, 150, 900, 300]  
                  macbook: [200, 300, 400, 350]  
                  smartphone: [400, 300, 450, 350]  
                  cup: [470, 300, 510, 350]  
                  box: [100, 250, 150, 300]  
                  water_bottles: [600, 350, 700, 450]   (gpt: 5.1954s | total: 5.1957s)
      ...
      ```


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
export OPENAI_API_KEY='sk-proj-...'
```

ImportError 발생 시
``` bash
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get install -y libgl1-mesa-glx libglib2.0-0
# pip install opencv-python timm einops flash_attn transformers
```

