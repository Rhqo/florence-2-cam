import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import matplotlib.font_manager as fm

import random
import re

from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Tuple, Optional

from openai import OpenAI
import io
import base64

import time

cap = cv2.VideoCapture(-1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

client = OpenAI()

# 한글 폰트
font_path = './NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

def run_example(task_prompt, text_input=None, pil_image=None):
    if text_input is None:    
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))
    return parsed_answer

def plot_bbox(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    thickness: int = 2,
    font_scale: float = 1,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    text_padding: int = 5
) -> np.ndarray:
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text_color_rgb = tuple(c / 255.0 for c in text_color[::-1])

    # Create a figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    # Plot each bounding box
    for idx, (bbox, label) in enumerate(zip(bboxes, labels if labels else [])):
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min

        # Generate a random color
        color = tuple(random.randint(0, 255) for _ in range(3))
        color_rgb = tuple(c / 255.0 for c in color[::-1])

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=thickness,
            edgecolor=color_rgb,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Annotate the label
        text = label if labels else str(idx)
        ax.text(
            x_min + text_padding,
            y_min - text_padding,
            text,
            fontsize=font_scale * 10,
            color=text_color_rgb,
            bbox=dict(facecolor=color_rgb, alpha=0.5, pad=text_padding),
            fontproperties=fontprop
        )

    ax.axis('off')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # Convert RGBA to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img_bgr

def parse_gpt_response(response):
    situation = ""
    bboxes = []
    labels = []
    
    # 상황 추출
    situation_match = re.search(r'{situation}:\s*(.*?)(?:\n|$)', response)
    if situation_match:
        situation = situation_match.group(1).strip()
    
    # "물체명: [x1, y1, x2, y2]"
    pattern1 = re.compile(r'([^:]+):\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
    for match in pattern1.finditer(response):
        label = match.group(1).strip()
        x_min, y_min, x_max, y_max = map(int, match.groups()[1:5])
        bboxes.append((x_min, y_min, x_max, y_max))
        labels.append(label)
    
    return situation, bboxes, labels

def process_gpt(system_prompt, user_prompt, input_image):
    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]
    
    # API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    gpt_response = response.choices[0].message.content
    
    # gpt response parsing
    situation, bboxes, labels = parse_gpt_response(gpt_response)
    
    # PIL -> opencv
    frame = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # bounding box
    if bboxes:
        annotated_image = plot_bbox(frame, bboxes, labels)
    else:
        annotated_image = frame
    
    return annotated_image, gpt_response

def main(system_prompt):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 현재 프레임 시작 시간
        frame_start_time = time.time()
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        result = run_example(task_prompt='<REGION_PROPOSAL>', pil_image=pil_image)
        bbox = result['<REGION_PROPOSAL>']['bboxes']
        bbox = sorted(bbox, key=lambda x: x[0])
        result_bbox = str(bbox)

        # GPT 처리
        gpt_start_time = time.time()
        annotated_image, gpt_response = process_gpt(system_prompt, result_bbox, pil_image)
        gpt_process_time = time.time() - gpt_start_time
        
        frame_process_time = time.time() - frame_start_time
        
        # gpt 응답 시간, 프레임 처리 시간 출력
        print(f'{gpt_response} (gpt: {gpt_process_time:.4f}s | total: {frame_process_time:.4f}s)')

        cv2.imshow("display", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

system_prompt = """유저의 입력 prompt로 당신은 2가지를 제공받습니다: 이미지와 bounding box 정보입니다.
                이미지와 bounding box를 분석한 후 다음 두 가지를 분석해주세요:
                    1. 이미지에 보이는 전체적인 상황
                    2. Bounding box 속 물체들의 캡션

                위 정보를 바탕으로 사진을 찍는 사람이 처한 상황을 다음 형식으로 작성해주세요:

                    {situation}: [제공된 이미지를 기반으로 한 상황 설명]
                    {location}: [위치 좌표는 [x_min, y_min, x_max, y_max] 형식으로 제공됨.]

                    상황 인식 답변 규칙:
                    - '나는'이라는 주어 사용 금지
                    - 10자 이내로 간결하게 설명
                    - '현대적인', '세련된' 등의 주관적 표현 사용 금지
                    - 감정이나 느낌은 제외하고 1인칭 시점으로 작성
                    - 모든 답변은 한국어로 작성
                    - 문장은 반드시 '~한 상황'으로 끝나야 함
                    - 설명은 관찰된 내용에 기반하여 사실적으로 작성

                    물체 정보 답변 규칙:
                    - 물체의 위치는 반드시 다음 형식으로 제공됨: [x_min, y_min, x_max, y_max]
                    - x_min이 오름차순이 되도록 정렬되어 제공되므로, x_min이 작은 순서대로 작성
                    - '고기'처럼 범용적인 표현 대신, 구체적인 물체명 사용
                    - 모든 답변은 한국어로 작성
                    - 물체가 여러 개인 경우 각 물체마다 한 줄에 작성

                    예시:
                    {situation}: 식당에서 밥을 먹는 상황
                    {location}: 피자: [100, 150, 200, 250]
                                파스타: [300, 200, 500, 350]

                이 형식과 규칙을 엄격히 준수하여 응답을 작성해주세요."""


# 메인 실행
if __name__ == "__main__":
    main(system_prompt)
