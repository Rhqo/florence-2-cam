import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import matplotlib.font_manager as fm

import random
import re

from typing import List, Tuple, Optional

from openai import OpenAI
import io
import base64

import time
import torchvision
import torchvision.transforms as T

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

client = OpenAI()

# 한글 폰트
font_path = './NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()


# Region Proposal: Faster R-CNN 적용
# Faster R-CNN 모델 로드
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()
faster_rcnn_model.to(device)

def run_region_proposal(pil_image: Image.Image) -> dict:
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(pil_image).to(device)
    with torch.no_grad():
        predictions = faster_rcnn_model([image_tensor])[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    threshold = 0.6  # score 임계치
    final_boxes = []
    for box, score in zip(boxes, scores):
        if score >= threshold:
            final_boxes.append(tuple(map(int, box)))
    
    # x_min 기준으로 정렬
    final_boxes = sorted(final_boxes, key=lambda b: b[0])
    
    return {'<REGION_PROPOSAL>': {'bboxes': final_boxes}}

def plot_bbox(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    thickness: int = 2,
    font_scale: float = 1,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    text_padding: int = 5
) -> Image.Image:
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text_color_rgb = tuple(c / 255.0 for c in text_color[::-1])

    # Create a figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    # bounding box
    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min

        # random color
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
        if labels and idx < len(labels):
            text = labels[idx]
            ax.text(
                x_min + text_padding,
                y_min - text_padding,
                text,
                fontsize=font_scale * 10,
                color=text_color_rgb,
                bbox=dict(facecolor=color_rgb, alpha=0.5, pad=text_padding),
                fontproperties=fontprop
            )
        # Annotate the label if 'no label'
        else:
            ax.text(
                x_min + text_padding,
                y_min - text_padding,
                f"객체 {idx+1}",
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
    return Image.fromarray(img).convert("RGB")

def parse_gpt_response(response, num_expected_bboxes):
    situation = ""
    labels = []
    
    # 상황 추출
    situation_match = re.search(r'{situation}:\s*(.*?)(?:\n|$)', response)
    if situation_match:
        situation = situation_match.group(1).strip()
    
    # 객체 캡션 추출, "객체 N: 설명" 형식
    location_section = re.search(r'{location}:(.*?)(?=$|\n\n)', response, re.DOTALL)
    if location_section:
        location_content = location_section.group(1).strip()

        for line in location_content.split('\n'):
            line = line.strip()
            if not line:
                continue

            object_match = re.match(r'(?:객체|Object)\s+(\d+):\s*(.*)', line, re.IGNORECASE)
            if object_match:
                labels.append(object_match.group(2).strip())
                continue
    
    # bounding box num == label num
    if len(labels) < num_expected_bboxes:
        for i in range(len(labels), num_expected_bboxes):
            labels.append(f"객체 {i+1}")
    elif len(labels) > num_expected_bboxes:
        labels = labels[:num_expected_bboxes]
    
    return situation, labels

def process_gpt(system_prompt, user_prompt, input_image, rcnn_bboxes):
    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    enhanced_prompt = (
        f"{user_prompt}\n\n"
        f"이미지에서 감지된 {len(rcnn_bboxes)}개의 객체가 있습니다. 이 객체들은 x_min(왼쪽 좌표)의 오름차순으로 정렬되어 있습니다. "
        f"반드시 {len(rcnn_bboxes)}개의 객체에 대한 설명을 순서대로 제공해주세요."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": enhanced_prompt},
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
    situation, labels = parse_gpt_response(gpt_response, len(rcnn_bboxes))
    
    # PIL -> opencv
    frame = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    annotated_image = plot_bbox(frame, rcnn_bboxes, labels)

    # bounding box
    return annotated_image, gpt_response, situation, labels

def main(system_prompt, user_prompt):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 현재 프레임 시작 시간
        frame_start_time = time.time()
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        result = run_region_proposal(pil_image)
        rcnn_bboxes = result['<REGION_PROPOSAL>']['bboxes']
        
        # GPT 처리
        gpt_start_time = time.time()
        annotated_image, gpt_response, situation, labels = process_gpt(system_prompt, user_prompt, pil_image, rcnn_bboxes)
        gpt_process_time = time.time() - gpt_start_time
        
        frame_process_time = time.time() - frame_start_time

        # gpt 응답 시간, 프레임 처리 시간 출력
        print(f'{gpt_response} (gpt: {gpt_process_time:.4f}s | total: {frame_process_time:.4f}s)')
        
        annotated_cv_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("display", annotated_cv_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

system_prompt = """유저의 입력 이미지에서 바운딩 박스로 표시된 객체들을 분석해주세요.
                    이미지와 객체 정보를 분석한 후 다음 두 가지를 작성해주세요:
                        1. 이미지에 보이는 전체적인 상황
                        2. 감지된 각 객체(바운딩 박스)에 대한 구체적인 설명

                    중요: 반드시 사용자가 알려준 정확한 객체 수만큼 설명을 작성해야 합니다. 더 많거나 적게 작성하지 마세요.

                    다음 형식으로 작성해주세요:

                        {situation}: [제공된 이미지를 기반으로 한 상황 설명]
                        {location}: 
                        객체 1: [첫 번째 객체 설명]
                        객체 2: [두 번째 객체 설명]
                        ...

                        상황 인식 답변 규칙:
                        - '나는'이라는 주어 사용 금지
                        - 10자 이내로 간결하게 설명
                        - '현대적인', '세련된' 등의 주관적 표현 사용 금지
                        - 감정이나 느낌은 제외하고 1인칭 시점으로 작성
                        - 모든 답변은 한국어로 작성
                        - 문장은 반드시 '~한 상황'으로 끝나야 함
                        - 설명은 관찰된 내용에 기반하여 사실적으로 작성

                        객체 설명 규칙:
                        - '고기'처럼 범용적인 표현 대신, 구체적인 물체명 사용
                        - 모든 답변은 한국어로 작성
                        - 정확히 사용자가 알려준 객체 수만큼 설명 작성 (누락 없이)
                        - 각 객체마다 한 줄에 작성
                        - '객체 N:' 형식으로 각 객체 설명 시작

                        예시:
                        {situation}: 식당에서 밥을 먹는 상황
                        {location}: 객체 1: 피자
                                    객체 2: 파스타

                    이 형식과 규칙을 엄격히 준수하여 응답을 작성해주세요."""

user_prompt = "이 이미지에서 보이는 상황과 각 객체를 분석해주세요."

if __name__ == "__main__":
    main(system_prompt, user_prompt)
