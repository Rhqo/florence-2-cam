import time
from collections import deque
from queue import Queue
from omegaconf import OmegaConf
from threading import Event, Lock, Thread
import random
import atexit
import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

from ollama import chat
from pydantic import BaseModel, Field
from typing import List, Literal

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None

# === configs == #
cfg = OmegaConf.load('cfg/webcam_cfg.yaml')
args = cfg.Args

# 한글 폰트 설정
font_path = './NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()


# ollama 모델 설정
# 가능한 모델 : 'gemma3:1b', 'gemma3', 'gemma3:12b', 'llama3.2-vision', ...
ollama_model = 'gemma3'


# ollama용 시스템 및 유저 프롬프트
system_prompt = """유저의 입력 이미지에서 빨간색 바운딩 박스로 표시된 객체들을 분석해주세요.
                이미지와 객체 정보를 분석한 후 다음 두 가지를 작성해주세요:
                    1. 이미지에 보이는 전체적인 상황
                    2. 감지된 각 객체(바운딩 박스)에 대한 구체적인 설명

                중요: 빨간색 바운딩 박스로 표시된 객체 수만큼 순서대로 설명을 작성해야 합니다. 더 많거나 적게 작성하지 마세요.

                {situation} 답변 규칙:
                - '나는'이라는 주어 사용 금지
                - 10자 이내로 간결하게 설명
                - '현대적인', '세련된' 등의 주관적 표현 사용 금지
                - 감정이나 느낌은 제외하고 1인칭 시점으로 작성
                - 모든 답변은 한국어로 작성
                - 문장은 반드시 '~한 상황'으로 끝나야 함
                - 설명은 관찰된 내용에 기반하여 사실적으로 작성

                {object} 설명 규칙:
                - '고기'처럼 범용적인 표현 대신, 구체적인 물체명 사용
                - 모든 답변은 한국어로 작성
                - 누락 없이 사용자가 알려준 객체 수만큼 순서대로 설명 작성
                - 각 객체마다 한 줄에 작성
                - '객체 N:' 형식으로 각 객체 설명 시작

                예시:
                {situation}: 식당에서 밥을 먹는 상황
                {object}: 객체 1: 피자
                            객체 2: 파스타

                이 형식과 규칙을 엄격히 준수하여 응답을 작성해주세요."""

user_prompt = "이 이미지에서 보이는 상황과 빨간색 바운딩 박스로 표시된 객체를 분석해주세요."

# Ollama 응답 형식
class ImageAnalysisResponse(BaseModel):
    situation: str = Field(..., description="전체적인 상황 설명")
    object: List[str] = Field(..., description="감지된 각 객체(바운딩 박스)에 대한 설명")

# 결과 저장 dictionary
# element: (boxes, labels, situation, ollama_response, processing_time, ts_processed)
detection_results = {
    'boxes': None,
    'labels': None,
    'situation': None,
    f'{ollama_model}_response': None,
    'processing_time': None,
    'ts_processed': None
}
detection_results_mutex = Lock()

# Faster R-CNN 모델 로드
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()
faster_rcnn_model.to(args.device)

# Faster R-CNN region proposal
def run_region_proposal(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(pil_image).to(args.device)
    
    with torch.no_grad():
        predictions = faster_rcnn_model([image_tensor])[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # 임계값으로 필터링
    threshold = 0.8  # 점수 임계값
    final_boxes = []
    for box, score in zip(boxes, scores):
        if score >= threshold:
            final_boxes.append(tuple(map(int, box)))
    
    # x_min 기준으로 정렬
    final_boxes = sorted(final_boxes, key=lambda b: b[0])
    
    return final_boxes

# Ollama 모델 사용
def process_ollama(frame, boxes):
    # 바운딩 박스가 있는 이미지 생성
    annotated_image = frame.copy()
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        text = f"obj_{i+1}"
        text_position = (x_min, y_min - 10)
        cv2.putText(
            annotated_image,       
            text,              
            text_position,         
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                    
            (0, 0, 255),          
            2                           
        )
    
    # 프레임을 PIL 이미지로 변환하고 임시 파일로 저장
    pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    temp_image_path = "temp_bbox_image.png"
    pil_image.save(temp_image_path)
    
    # 프롬프트 준비
    enhanced_prompt = (
        f"{user_prompt}\n\n"
        f"이미지에서 감지된 {len(boxes)}개의 객체가 있습니다. 이 객체들은 x_min(왼쪽 좌표)의 오름차순으로 정렬되어 있습니다. "
        f"반드시 {len(boxes)}개의 객체에 대한 설명을 순서대로 제공해주세요."
    )
    
    try:
        # Ollama API 호출
        response = chat(
            model=f'{ollama_model}',
            format=ImageAnalysisResponse.model_json_schema(),
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': enhanced_prompt,
                    'images': [temp_image_path],
                },
            ],
            options={'temperature': 0},
        )
        
        # 응답 파싱
        analysis = ImageAnalysisResponse.model_validate_json(response.message.content)
        
        # 상황 및 라벨 추출
        situation = analysis.situation
        labels = analysis.object
        
        # formatting
        formatted_response = f"{{situation}}: {situation}\n{{object}}: \n"
        for i, label in enumerate(labels):
            formatted_response += f"객체 {i+1}: {label}\n"
        
        # 임시 파일 정리
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return formatted_response, situation, labels
        
    except Exception as e:
        print(f"Error with Ollama API: {e}")
        # 오류 발생 시 기본값 반환
        return "API error", "분석 실패", [f"객체 {i+1}" for i in range(len(boxes))]

# plot_bbox 
def plot_bbox(
    image,
    bboxes,
    labels=None,
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

        color = tuple(random.randint(0, 255) for _ in range(3))
        color_rgb = tuple((1,0,0))

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

# 종료 시 임시 파일 제거
def cleanup_temp_files():
    temp_image_path = "temp_bbox_image.png"
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

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
            
            # 1. Faster R-CNN region proposal 수행
            boxes = run_region_proposal(frame)
            
            # 객체가 감지되지 않은 경우 건너뛰기
            if not boxes:
                print("No objects detected")
                event_inference_done.set()
                continue
                
            # 2. Ollama로 상황 및 객체 설명 요청
            ollama_start_time = time.time()
            ollama_response, situation, labels = process_ollama(frame, boxes)
            ollama_process_time = time.time() - ollama_start_time
            
            # 총 처리 시간
            total_process_time = time.time() - start_time
            
            # 3. display thread를 위한 결과 저장
            with detection_results_mutex:
                detection_results['boxes'] = boxes
                detection_results['labels'] = labels
                detection_results['situation'] = situation
                detection_results[f'{ollama_model}_response'] = ollama_response
                detection_results['processing_time'] = {
                    'ollama': ollama_process_time,
                    'total': total_process_time
                }
                detection_results['ts_processed'] = time.time()
            
            print(f"{ollama_response} ({ollama_model}: {ollama_process_time:.4f}s | total: {total_process_time:.4f}s)")
            
            # 4. synchronous mode를 위한 감지 완료 신호
            event_inference_done.set()
            
            _ts = time.time()
            if _ts_last is not None and _ts - _ts_last < min_interval:
                time.sleep(min_interval - (_ts - _ts_last))
            _ts_last = time.time()
            
        except Exception as e:
            print(f"Error in inference_detection: {e}")
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
                    
                    # 현재 프레임에 바운딩 박스 그리기
                    annotated_image = plot_bbox(
                        image=frame, 
                        bboxes=boxes, 
                        labels=labels,
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
    atexit.register(cleanup_temp_files)
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
