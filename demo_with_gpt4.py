import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid display errors
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import random
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
import json

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
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
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

def process_images(input_dir, output_dir, task_prompt):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of filenames and sort them in ascending order
    filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    results = {}

    # Open the text file for writing results
    with open(os.path.join(output_dir, "florence_results.txt"), "w") as f:
        # Use tqdm to display a progress bar
        for filename in tqdm(filenames, desc="Processing images"):
            input_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output.png")
            
            # Load the image
            pil_image = Image.open(input_path)
            image_np = np.array(pil_image)

            # Run the example
            task_prompt_0 = task_prompt
            result_0 = run_example(task_prompt=task_prompt_0, pil_image=pil_image)

            task_prompt_1 = "<CAPTION_TO_PHRASE_GROUNDING>"
            result_1 = run_example(task_prompt=task_prompt_1, text_input=result_0[task_prompt_0], pil_image=pil_image)

            # Save the results to the dictionary
            results[filename] = {
                "caption": result_0[task_prompt_0],
                "labels": result_1[task_prompt_1]['labels']
            }

            # Draw bounding boxes on the image
            output_image = plot_bbox(image_np, result_1[task_prompt_1]['bboxes'], result_1[task_prompt_1]['labels'])
            output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            # Save the output image
            cv2.imwrite(output_image_path, output_rgb)

            # Save the results to the text file
            f.write(f"===== {filename} Result =====\n")
            f.write(str(result_0))
            f.write("\n\n")
            f.write(str(result_1[task_prompt_1]['labels']))
            f.write("\n\n")

    return results

def process_text(system_prompt, results, output_dir):
    response_counter = 1
    for result in tqdm(results, desc="Processing text"):
        system_prompt = system_prompt
        user_prompt = json.dumps(results[result])
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = completion.choices[0].message.content

        # Save the results to the text file
        with open(os.path.join(output_dir, "results.txt"), "a") as f:
            f.write(f"{response_counter}. ")
            f.write(str(response))
            f.write("\n\n")
        
        response_counter += 1


def main(input_dir, output_dir, task_prompt, system_prompt):
    results = process_images(input_dir, output_dir, task_prompt)
    process_text(system_prompt, results, output_dir)


# system_prompt = """당신은 이미지에 대한 설명을 제공받고 이것을 정형 데이터로 변환해주는 데이터 엔지니어입니다.
#                 관찰자 시점으로 적혀있는 설명을 보고 사진을 찍은 사람이 어떤 상황에 처해져 있는지, 보이는 객체들은 무엇인지 1인칭의 시점으로 설명해주세요.
#                 그리고 이것을 상황과 객체로 분류해서 정형화해주세요.
#                 모든 답변은 한국어를 사용해서 답변해주세요.

#                 다음과 같은 구조화된 형식으로 응답을 작성해주세요:
#                 {
#                     "상황": "장면 또는 상황에 대한 설명을 자연어로 작성해주세요.",
#                     "객체": [
#                         {
#                         "이름": "객체의 이름",
#                         "위치": "객체의 위치에 대한 상세 정보",
#                         "속성": "객체의 속성이나 특징"
#                         },
#                         ...
#                     ]
#                 }
#                 이 형식을 엄격히 준수하며 응답을 작성해주세요."""

# system_prompt = """관찰자 시점으로 적혀있는 설명을 보고 사진을 찍는 사람이 처해진 상황에 대해서 감정이나 느낌은 제외하고 1인칭 시점의 한 문장으로 짧게 설명해 주세요.
#                    모든 답변은 한국어를 사용해서 답변해 주세요."""

system_prompt = """관찰자 시점으로 적혀있는 설명을 보고 사진을 찍는 사람이 처한 상황에 대해서 감정이나 느낌은 제외하고 1인칭 시점으로 10자 이내로 설명해 주세요.
                   모든 답변은 한국어를 사용해서 답변해 주세요.
                   
                   다음은 답변에 대한 규칙입니다:
                    - '나는'이라는 주어를 사용하면 안 됩니다.
                    - '현대적인', '세련된' 등의 주관적인 표현을 사용하면 안 됩니다.
                    - 문장의 끝은 '~한 상황'으로 끝나야 합니다.
                    - 모든 답변은 설명을 기반으로 거짓없이 작성되어야 합니다.

                   이 규칙을 엄격히 준수하며 응답을 작성해주세요."""

input_dir = "/workspace/volume/input_situation(modified)"
output_dir = "/workspace/volume/situation_output_10chars_2"
task_prompt = "<MORE_DETAILED_CAPTION>" # "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"


main(input_dir, output_dir, task_prompt, system_prompt)
