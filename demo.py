import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


def run_example(task_prompt, text_input=None, pil_image=None):
    if text_input is None:    
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt,images=pil_image,return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))

    return parsed_answer


def draw_bboxes(image, bboxes, labels):

    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    label_text = label    
    cv2.putText(image, label_text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

    return image


while True:
    ret, frame = cap.read()

    pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    ### Object Detection ###
    # result = run_example(task_prompt="<OD>", pil_image=pil_image)
    # bboxes = result['<OD>']['bboxes']
    # labels = result['<OD>']['labels']
    # print(labels)


    ### Caption to Phase Grounding ###
    # '<CAPTION>' / '<DETAILED_CAPTION>' / '<MORE_DETAILED_CAPTION>'
    task_caption = '<CAPTION>'
    result = run_example(task_prompt=task_caption, pil_image=pil_image)
    text_input = result[task_caption]
    result = run_example(task_prompt='<CAPTION_TO_PHRASE_GROUNDING>', text_input=text_input, pil_image=pil_image)
    result[task_caption] = text_input


    bboxes = result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    labels = result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
    print(labels)
    # print(result[task_caption])

    annotated_image = draw_bboxes(frame.copy(), bboxes, labels)

    cv2.imshow("display", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()