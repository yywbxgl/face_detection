import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.backend as backend
import time
import sys,os

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def postprocess(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480)) 
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img


def img_inference(img_ori):
    # preprocess img 
    img = preprocess(img_ori)
    
    # inference
    onnx_model = onnx.load('models/ultra_light_640_optimized.onnx')
    # onnx_model = onnx.load('models/ultra_light_640.onnx')
    ort_session = backend.prepare(onnx_model)
    confidences, boxes = ort_session.run(img)
    
    # postprogcess
    h = img_ori.shape[0]
    w = img_ori.shape[1]
    boxes, labels, probs = postprocess(w, h, confidences, boxes, 0.5)
    
    return boxes, labels, probs 



def demo_video():
    # load the model, create runtime session & get input variable name
    onnx_model = onnx.load('models/ultra_light_640_optimized.onnx')
    # onnx_model = onnx.load('models/ultra_light_640.onnx')
    ort_session = backend.prepare(onnx_model)

    video_capture = cv2.VideoCapture(0)
    # LIVE_URL = "rtsp://admin:admin123@172.16.1.29/cam/realmonitor?channel=1&subtype=0"
    # video_capture = cv2.VideoCapture(LIVE_URL)
    print("Video fps:", video_capture.get(cv2.CAP_PROP_FPS))
    
    start_time = endtime = time.time()
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        h, w, _ = frame.shape
        frame_count += 1

        # preprocess
        img = preprocess(frame)
        # inference
        confidences, boxes = ort_session.run(img)
        # postprocess 
        boxes, labels, probs = postprocess(w, h, confidences, boxes, 0.6)

        # draw boxes
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
            # cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # text = "face: {labels[i]}"
            # cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!

        if frame_count == 10:
            endtime = time.time()
            print("\rReally fps=%s "%(frame_count/(endtime-start_time)), end="")
            start_time = endtime
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    

def demo_image():

    img_file = "imgs/test2.jpg"
    if len(sys.argv) > 1:
        img_file =  sys.argv[1]
    
    img_ori = cv2.imread(img_file)
    
    # img inference
    boxes, labels, probs = img_inference(img_ori)
    
    # draw boxes
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        cv2.rectangle(img_ori, (x1, y1), (x2, y2), (80,18,236), 2)

    cv2.imshow('Video', img_ori)
    cv2.waitKey(0) 


    
if __name__ == '__main__':
    demo_video()
    # demo_image()