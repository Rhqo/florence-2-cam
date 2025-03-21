import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt  
import matplotlib.patches as patches

import random
import re

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

client = OpenAI()

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

        # Create a Rectangle patch
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
            bbox=dict(facecolor=color_rgb, alpha=0.5, pad=text_padding)
        )

    # Remove axis ticks and labels
    ax.axis('off')

    # Draw the canvas and retrieve the image as RGBA buffer
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
    
    # PIL -> opencv 형식으로 변환
    frame = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # bounding box
    if bboxes and labels:
        annotated_image = plot_bbox(frame, bboxes, labels)
    else:
        annotated_image = frame
    
    return annotated_image, gpt_response

def main(user_prompt, system_prompt):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_start_time = time.time()
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # GPT 처리
        gpt_start_time = time.time()
        annotated_image, gpt_response = process_gpt(system_prompt, user_prompt, pil_image)
        gpt_process_time = time.time() - gpt_start_time
        
        current_time = time.time()
        frame_process_time = current_time - frame_start_time
        
        # gpt 응답 시간, 프레임 처리 시간 출력
        print(f'{gpt_response} (gpt: {gpt_process_time:.4f}s | total: {frame_process_time:.4f}s)')

        cv2.imshow("display", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# System Prompt
system_prompt = """이미지를 분석한 후 다음 두 가지를 분석해주세요:
                1. 이미지에 보이는 전체적인 상황
                2. 주요 물체들의 위치

                위 정보를 바탕으로 사진을 찍는 사람이 처한 상황을 다음 형식으로 작성해주세요:

                {situation}: [사진을 찍는 사람이 처한 상황을 10자 이내로 간결하게 설명. 감정이나 느낌은 제외하고 1인칭 시점으로 작성.]
                {location}: [위치 좌표는 반드시 [x_min, y_min, x_max, y_max] 형식으로 작성]

                상황 인식 답변 규칙:
                - '나는'이라는 주어 사용 금지
                - '현대적인', '세련된' 등의 주관적 표현 사용 금지
                - 모든 답변은 한국어로 작성
                - 문장은 반드시 '~한 상황'으로 끝나야 함
                - 설명은 관찰된 내용에 기반하여 사실적으로 작성

                위치 정보 답변 규칙:
                - 물체의 위치는 반드시 다음 형식으로 제공: 물체명: [x_min, y_min, x_max, y_max]
                - meat 처럼 범용적인 표현 대신, 구체적인 물체명 사용
                - 물체는 영어로 작성
                - 물체가 여러 개인 경우 각 물체마다 한 줄에 작성


                예시:
                {situation}: 식당에서 밥을 먹는 상황
                {location}: pizza: [100, 150, 200, 250]
                            pasta: [300, 200, 500, 350]

                이 형식과 규칙을 엄격히 준수하여 응답을 작성해주세요."""

# User Prompt
user_prompt = "이 이미지에서 보이는 상황과 물체들의 위치를 분석해주세요."

if __name__ == "__main__":
    main(user_prompt, system_prompt)
