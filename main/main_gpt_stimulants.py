import time
from collections import deque
from queue import Queue
from omegaconf import OmegaConf
from threading import Event, Lock, Thread
import random
import io
import base64

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

from openai import OpenAI
from pydantic import BaseModel

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None

# === configs == #
cfg = OmegaConf.load('cfg/webcam_cfg.yaml')
args = cfg.Args

# OpenAI client
client = OpenAI()

# 한글 폰트 설정
font_path = './NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

# Structured output을 위한 Pydantic 모델 정의
class ObjectAttribute(BaseModel):
    korean: str
    english: str
    used_by: str  # "camera_wearer" or "other_person" or "background"
    is_stimulant: bool = False

class FreeDetectionResponse(BaseModel):
    situation: str
    objects: list[ObjectAttribute]

# GPT용 시스템 프롬프트
system_prompt = """이미지를 분석하여 다음을 제공하세요:

1. situation: 전체 상황 (15자 이내, '~한 상황'으로 끝남, 한국어)
2. objects: 주요 객체 (한국어, 영어, used_by, is_stimulant)

**중요: 이 이미지는 카메라 착용자의 1인칭 시점입니다.**

각 객체마다 다음을 판단:
- korean: 객체 이름 (한국어)
- english: 객체 이름 (영어)
- used_by: 누가 사용/접근하는가?
  * "camera_wearer": 카메라 착용자가 직접 사용 (손에 들고 있거나, 바로 앞에서 사용 중)
  * "other_person": 다른 사람이 사용 (다른 사람 손에 있거나, 다른 사람이 접근 중)
  * "background": 배경에 있음 (아무도 사용하지 않음)
- is_stimulant: 일주기 리듬 영향 요소인가?
  * 커피, 에너지드링크, 모니터, 휴대폰, TV, 음식, 술, 담배 등 → true
  * 가구, 일반 물건 → false

**중요: is_stimulant=true는 used_by="camera_wearer"인 경우에만 의미가 있습니다.**

예시 1 (내가 커피를 마시는 중):
{"situation": "커피를 마시는 상태", "objects": [
    {"korean": "커피", "english": "coffee", "used_by": "camera_wearer", "is_stimulant": true},
    {"korean": "소파", "english": "sofa", "used_by": "background", "is_stimulant": false}
]}

예시 2 (내가 모니터로 작업 중):
{"situation": "책상에서 작업하는 상태", "objects": [
    {"korean": "모니터", "english": "monitor", "used_by": "camera_wearer", "is_stimulant": true},
    {"korean": "키보드", "english": "keyboard", "used_by": "camera_wearer", "is_stimulant": false}
]}

예시 3 (다른 사람이 커피 마시는 것을 보는 중):
{"situation": "다른 사람을 보는 상황", "objects": [
    {"korean": "커피", "english": "coffee", "used_by": "other_person", "is_stimulant": false},
    {"korean": "테이블", "english": "table", "used_by": "background", "is_stimulant": false}
]}"""

user_prompt = "이 이미지에서 **카메라 착용자(나)가 직접 사용하고 있는** 객체들과 상황을 설명해주세요. 다른 사람이 사용하는 것은 제외하세요."

# 결과 저장 dict
detection_results = {
    'boxes': None,
    'labels': None,
    'is_stimulants': None,
    'situation': None,
    'gpt_response': None,
    'processing_time': None,
    'ts_processed': None
}
detection_results_mutex = Lock()

# Grounding DINO 모델 로드
print("Loading Grounding DINO model...")
grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
grounding_model.to(args.device)
grounding_model.eval()
print("Grounding DINO model loaded!")

# Grounding DINO로 객체 grounding
def run_grounding(frame, english_queries, korean_labels, is_stimulants, threshold=0.25, text_threshold=0.20):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Grounding DINO Format: "query1 . query2 . query3"
    text_prompt = " . ".join(english_queries)

    # Grounding DINO 입력 생성
    inputs = grounding_processor(images=pil_image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Post-processing
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]]  # (height, width)
    )[0]

    # Parsing
    boxes = []
    labels = []
    stimulants = []

    # 각 쿼리별로 가장 confidence 높은 box 선택
    for eng_query, kor_label, is_stim in zip(english_queries, korean_labels, is_stimulants):
        best_box = None
        best_score = 0

        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"]
        ):
            if eng_query.lower() in label.lower() or label.lower() in eng_query.lower():
                if score > best_score:
                    best_score = score
                    best_box = box

        # Best box가 있으면 추가 (+ is_stimulant)
        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)
            boxes.append((x_min, y_min, x_max, y_max))
            labels.append(kor_label)
            stimulants.append(is_stim)

    return boxes, labels, stimulants

# GPT Vision API 호출
def process_gpt(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
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

    # API 호출 (Structured Outputs)
    response = client.beta.chat.completions.parse(
        model="gpt-5-nano",
        messages=messages,
        response_format=FreeDetectionResponse,
        reasoning_effort="minimal"
    )

    # Structured output parsing
    parsed_response = response.choices[0].message.parsed

    situation = parsed_response.situation
    object_pairs = parsed_response.objects

    # Post-processing: used_by가 "camera_wearer"가 아니면 is_stimulant를 False로 강제
    korean_descriptions = []
    english_descriptions = []
    is_stimulants = []

    for obj in object_pairs:
        korean_descriptions.append(obj.korean)
        english_descriptions.append(obj.english)

        # 카메라 착용자가 사용하는 경우에만 is_stimulant 유지, 그 외는 False
        if obj.used_by == "camera_wearer":
            is_stimulants.append(obj.is_stimulant)
        else:
            is_stimulants.append(False)

    # 응답 요약 (각성제는 [!] 표시)
    obj_display = [f"{k}[!]" if stim else k for k, stim in zip(korean_descriptions, is_stimulants)]
    gpt_response = f"{{situation}}: {situation}\n{{objects}}: " + ", ".join(obj_display)

    return gpt_response, situation, korean_descriptions, english_descriptions, is_stimulants

# plot_bbox
def plot_bbox(
    image,
    bboxes,
    labels=None,
    is_stimulants=None,
    thickness=2,
    font_scale=0.8,
    text_color=(0, 0, 0),
    text_padding=5
):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_copy = image.copy()

    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    text_color_rgb = tuple(c / 255.0 for c in text_color[::-1])

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    # draw bbox
    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min

        # is_stimulant에 따라 색상 결정: True=빨간색, False=파란색
        if is_stimulants and idx < len(is_stimulants) and is_stimulants[idx]:
            color_rgb = (1, 0, 0)  # 빨간색 (stimulant)
        else:
            color_rgb = (0, 0, 1)  # 파란색 (non-stimulant)

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=thickness,
            edgecolor=color_rgb,
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"객체 {idx+1}"
        if labels and idx < len(labels):
            label_text = labels[idx]

        ax.text(
            x_min + text_padding,
            y_min - text_padding,
            label_text,
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

    # RGBA를 RGB로 변환하고 PIL 이미지로 반환
    return Image.fromarray(img).convert("RGB")

# ------------------------------ #
# == Open-Voca Obj. Detection == #
# ------------------------------ #
def inference_detection():
    print("Thread 'OVD' started")
    min_interval = 1.0 / args.inference_fps
    _ts_last = None

    global detection_results

    while not event_exit.is_set():
        try:
            # 입력 대기
            while len(input_queue) < 1:
                time.sleep(0.001)
                if event_exit.is_set():
                    return

            with input_queue_mutex:
                ts_input, frame = input_queue.popleft()

            start_time = time.time()

            # 1. GPT로 상황 및 객체 자유 설명 요청 (한국어 + 영어 + is_stimulant)
            gpt_start_time = time.time()
            gpt_response, situation, korean_descriptions, english_descriptions, is_stimulants = process_gpt(frame)
            gpt_process_time = time.time() - gpt_start_time

            # 객체 설명이 없으면 건너뛰기
            if not korean_descriptions or not english_descriptions:
                print("No objects described by GPT")
                event_inference_done.set()
                continue

            # 2. Grounding DINO로 각 객체 grounding (영어 쿼리 사용, 한국어 라벨 + is_stimulant 반환)
            grounding_start_time = time.time()
            boxes, labels, stimulants = run_grounding(frame, english_descriptions, korean_descriptions, is_stimulants)
            grounding_process_time = time.time() - grounding_start_time

            # Grounding된 객체가 없으면 건너뛰기
            if not boxes:
                print("No objects grounded by Grounding DINO")
                event_inference_done.set()
                continue

            # 총 처리 시간
            total_process_time = time.time() - start_time

            # 3. display thread를 위한 결과 저장
            with detection_results_mutex:
                detection_results['boxes'] = boxes
                detection_results['labels'] = labels
                detection_results['is_stimulants'] = stimulants
                detection_results['situation'] = situation
                detection_results['gpt_response'] = gpt_response
                detection_results['processing_time'] = {
                    'gpt': gpt_process_time,
                    'grounding': grounding_process_time,
                    'total': total_process_time
                }
                detection_results['ts_processed'] = time.time()

            print(f"{gpt_response}")
            print(f"(gpt: {gpt_process_time:.4f}s | grounding: {grounding_process_time:.4f}s | total: {total_process_time:.4f}s)\n")

            # 4. synchronous mode를 위한 감지 완료 신호
            event_inference_done.set()

            _ts = time.time()
            if _ts_last is not None and _ts - _ts_last < min_interval:
                time.sleep(min_interval - (_ts - _ts_last))
            _ts_last = time.time()

        except Exception as e:
            print(f"Error in inference_detection: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

# ------------------- #
# == Video display == #
# ------------------- #
def display():
    print('Thread "display" started')

    while not event_exit.is_set():
        try:
            # 버퍼에서 새 프레임 가져오기
            ts_input, frame = frame_buffer.get()

            if ts_input is None:
                break

            # 현재 프레임 복사
            display_frame = frame.copy()

            # 감지 결과가 있으면 현재 프레임에 최신 바운딩 박스 적용
            with detection_results_mutex:
                if detection_results['boxes'] is not None:
                    boxes = detection_results['boxes']
                    labels = detection_results['labels']
                    is_stimulants = detection_results['is_stimulants']

                    # 현재 프레임에 바운딩 박스 그리기
                    annotated_image = plot_bbox(
                        image=frame,
                        bboxes=boxes,
                        labels=labels,
                        is_stimulants=is_stimulants,
                        thickness=2,
                        font_scale=1,
                        text_color=(0, 0, 0),  # 검정색 텍스트
                        text_padding=5
                    )
                    display_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

            # 화면에 표시
            cv2.imshow('webcam', display_frame)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input in (27, ord('q'), ord('Q')):
                event_exit.set()
                break

        except Exception as e:
            print(f"Error in display: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()
    event_exit.set()

# ----------------------- #
# == Extracting frames == #
# ----------------------- #
def read_camera():
    print(f"Thread 'input' started")
    cam_id = args.cam_id

    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()

    while not event_exit.is_set():
        # 카메라 프레임 캡처
        ret_val, frame = vid_cap.read()
        if ret_val:
            # 프레임을 입력 큐에 넣기
            ts_input = time.time()

            event_inference_done.clear()
            with input_queue_mutex:
                input_queue.append((ts_input, frame))

            if args.synchronous_mode:
                # 감지가 완료될 때까지 대기
                event_inference_done.wait()

            frame_buffer.put((ts_input, frame))  # 프레임을 버퍼에 넣기
        else:
            # 입력 종료 신호
            frame_buffer.put((None, None))
            break
    vid_cap.release()

def main():
    global frame_buffer
    global input_queue, input_queue_mutex
    global event_exit, event_inference_done

    # frame buffer size 설정
    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    # input queue 설정
    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    try:
        event_exit = Event()
        event_inference_done = Event()
        t_input = Thread(target=read_camera, args=())
        t_det = Thread(target=inference_detection, args=(), daemon=True)

        t_input.start()  # 프레임 읽기 스레드 시작
        t_det.start()    # 감지 스레드 시작

        display()  # 메인 스레드에서 디스플레이 실행
        t_input.join()  # 입력 스레드 조인 (non-daemon)

    except KeyboardInterrupt:
        event_exit.set()

if __name__ == "__main__":
    main()
