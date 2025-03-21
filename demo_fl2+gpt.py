import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt  
import matplotlib.patches as patches

import random

from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Tuple, Optional

from openai import OpenAI
import json
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

client = OpenAI()

# Define the run_example function
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

def process_text(system_prompt, results):
    
    user_prompt = json.dumps(results)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    response = completion.choices[0].message.content
    
    return response
    

def main(task_prompt, system_prompt):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 현재 프레임 시작 시간
        frame_start_time = time.time()
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        task_caption = task_prompt
        result = run_example(task_prompt=task_caption, pil_image=pil_image)
        text_input = result[task_caption]
        result = run_example(task_prompt='<CAPTION_TO_PHRASE_GROUNDING>', text_input=text_input, pil_image=pil_image)
        result[task_caption] = text_input

        bboxes = result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        labels = result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
        
        annotated_image = plot_bbox(frame.copy(), bboxes, labels)

        # GPT 처리
        gpt_start_time = time.time()
        gpt_response = process_text(system_prompt, result)
        gpt_process_time = time.time() - gpt_start_time
        
        # 현재 프레임 처리 완료 시간
        current_time = time.time()
        frame_process_time = current_time - frame_start_time
        
        print(f'{gpt_response} (gpt: {gpt_process_time:.4f}s | total: {frame_process_time:.4f}s)')

        cv2.imshow("display", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# 시스템 프롬프트 설정
system_prompt = """관찰자 시점으로 적혀있는 설명을 보고 사진을 찍는 사람이 처한 상황에 대해서 감정이나 느낌은 제외하고 1인칭 시점으로 10자 이내로 설명해 주세요.
                모든 답변은 한국어를 사용해서 답변해 주세요.

                다음은 답변에 대한 규칙입니다:
                - '나는'이라는 주어를 사용하면 안 됩니다.
                - '현대적인', '세련된' 등의 주관적인 표현을 사용하면 안 됩니다.
                - 문장의 끝은 '~한 상황'으로 끝나야 합니다.
                - 모든 답변은 설명을 기반으로 거짓없이 작성되어야 합니다.

                이 규칙을 엄격히 준수하며 응답을 작성해주세요."""

# 디렉토리 및 태스크 프롬프트 설정
task_prompt = "<MORE_DETAILED_CAPTION>"  # "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"

# 메인 실행
if __name__ == "__main__":
    main(task_prompt, system_prompt)
