import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.backend as backend
import time
import sys,os

import ultra_light


def Normalize(data):
    '''该function仅用于模仿 torh.nn.functional.normalize(data, p=2, dim=1)'''
    denom = [np.linalg.norm(data[i]) for i in range(len(data))]
    norm_data = [data[i] / denom[i] for i in range(len(data))]
    return np.array(norm_data)

def preprocess(img):
    if type(img) == str:
        img = cv2.imread(img)
    mean_vec = np.array([123.68, 116.28, 103.53])
    stddev_vec = np.array([57.6, 57.6, 57.6])
    img = cv2.resize(img, (160,160))
    img = np.transpose(img, (2, 0, 1))
    norm_img_data = np.zeros(img.shape).astype('float32')
    for i in range(img.shape[0]):
        norm_img_data[i, :, :] = (img[i, :, :] - mean_vec[i]) / stddev_vec[i]
    norm_img_data = np.expand_dims(norm_img_data, axis=0).astype(np.float32)
    return norm_img_data


def preprocess2(img):
    if type(img) == str:
        img = cv2.imread(img)
    # print(img.shape)
    img = cv2.resize(img, (160,160))
    img = np.transpose(img, (2, 0, 1))
    norm_img_data = np.zeros(img.shape).astype('float32')
    for i in range(img.shape[0]):
        norm_img_data[i, :, :] = (img[i, :, :] - 127.5) / 128
    norm_img_data = np.expand_dims(norm_img_data, axis=0).astype(np.float32)
    return norm_img_data

    
def postprocess(features):
    features = Normalize(features)
    return  features


def img_inference(img_ori):
    # preprocess img 
    img = preprocess2(img_ori)
    
    # inference
    onnx_model = onnx.load('models/FaceNet_vggface2_optmized.onnx')
    ort_session = backend.prepare(onnx_model)
    features = ort_session.run(img)
    
    # postprogcess
    features = np.array(features[0])
    features = postprocess(features)

    return features


def prepare_features(video_dir):
    files_list = os.listdir(video_dir)
    feature_list = []

    for label in files_list:
        print("start collecting faces from %s's data"%(label))
        cap = cv2.VideoCapture(os.path.join(video_dir, label))
        
        frame_count = 0
        img_count = 0
        while True:
            # read video frame
            ret, img_ori = cap.read()

            # process every 5 frames
            if frame_count % 25 == 0 and img_ori is not None:

                # face detection
                boxes, labels, probs = ultra_light.img_inference(img_ori)

                if boxes.shape[0] > 0:
                    x1, y1, x2, y2 = boxes[0,:]
                    # convert to gray scale image
                    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
                    # align and resize
                    # aligned_face = fa.align(raw_img, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                    # aligned_face = cv2.resize(aligned_face, (112,112))
                    # write to file
                    
                    crop_img = img_ori[y1:y2, x1:x2]
                    img_count += 1
                    cv2.imwrite('video_data/faces/%s_%s.jpg'%(label.split(".")[0], img_count), crop_img)

                    feature = img_inference(crop_img)
                    feature_list.append(feature)

                
            frame_count += 1
            # if video end
            if ret == False:
                break

    feature_list = np.array(feature_list)  
    np.save("./features.npy", feature_list)

    feature_list_name =  os.listdir("video_data/faces/")
    f = open("./features.txt", "w+")
    for i in feature_list_name:
        f.writelines(i + "\r\n")
    f.close()

    print("save features.")


def prepare_test_img(video_dir):
    files_list = os.listdir(video_dir)
    feature_list = []

    for label in files_list:
        print("start collecting faces from %s's data"%(label))
        cap = cv2.VideoCapture(os.path.join(video_dir, label))
        frame_count = 0
        while True:
            # read video frame
            ret, img_ori = cap.read()
            if ret == False:
                break
            else:
                frame_count += 1

            # process every 5 frames
            if frame_count % 100 == 0 and img_ori is not None:
                file_name = './video_data/capture/%s_%s.jpg'%(label.split(".")[0],frame_count)
                print("save img ", file_name)
                cv2.imwrite(file_name,  img_ori)



def demo_image():
    img_file = "video_data/capture/caijin_100.jpg"
    if len(sys.argv) > 1:
        img_file =  sys.argv[1]
        
    img_ori = cv2.imread(img_file)
    # print(img_ori.shape)
    
    # img inference
    boxes, labels, probs = ultra_light.img_inference(img_ori)
    
    # onnx prepare
    onnx_model = onnx.load('models/FaceNet_vggface2_optmized.onnx')
    ort_session = backend.prepare(onnx_model)
        
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        predictions = ""

        # draw boxes
        crop_img = img_ori[y1:y2, x1:x2]
        
        # facenet inference
        img = preprocess2(crop_img)
        features1 = ort_session.run(img)
        features1 = np.array(features1[0])
        features1 = postprocess(features1)

        # compare
        names = open("./features.txt").readlines()
        features_list = np.load("./features.npy").squeeze()
        diff = np.subtract(features_list, features1)
        dist = np.sum(np.square(diff), axis=1)
        idx = np.argmin(dist)
        print(dist)

        if dist[idx] < 1:
            predictions = names[idx].split("_")[0]
            print(dist[idx], predictions, idx)
        else:
            predictions = "unknown"
            print(predictions)

        # draw img
        cv2.rectangle(img_ori, (x1, y1), (x2, y2), (80,18,236), 2)
        cv2.rectangle(img_ori, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = predictions
        print(text)
        cv2.putText(img_ori, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)


    cv2.imshow('Video', img_ori)
    cv2.waitKey(0) 
    
    
def demo_video():

    video_file = "video_data/chenyiyi.avi"
    if len(sys.argv) > 1:
        video_file =  sys.argv[1]
    
    # load the model, create runtime session & get input variable name
    onnx_model = onnx.load('models/ultra_light_640_optimized.onnx')
    # onnx_model = onnx.load('models/ultra_light_640.onnx')
    detection_session = backend.prepare(onnx_model)

    # onnx prepare
    facenet_model = onnx.load('models/FaceNet_vggface2_optmized.onnx')
    facenet_session = backend.prepare(facenet_model)

    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture(video_file)
    # LIVE_URL = "rtsp://admin:admin123@172.16.1.29/cam/realmonitor?channel=1&subtype=0"
    # video_capture = cv2.VideoCapture(LIVE_URL)
    print("Video fps:", video_capture.get(cv2.CAP_PROP_FPS))


    names = open("./features.txt").readlines()
    features_list = np.load("./features.npy").squeeze()

    start_time = endtime = time.time()
    frame_count = 0

    while True:
        ret, img_ori = video_capture.read()
        # if ret == False:
        #     break
        if img_ori is None:
            continue

        h, w, _ = img_ori.shape
        frame_count += 1

        # preprocess
        img = ultra_light.preprocess(img_ori)
        # inference
        confidences, boxes = detection_session.run(img)
        # postprocess 
        boxes, labels, probs = ultra_light.postprocess(w, h, confidences, boxes, 0.6)
    
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            predictions = ""

            # draw boxes
            crop_img = img_ori[y1:y2, x1:x2]
            if crop_img.shape[0]==0 or crop_img.shape[1]==0:
                continue
            
            # facenet inference
            img = preprocess2(crop_img)
            features1 = facenet_session.run(img)
            features1 = np.array(features1[0])
            features1 = postprocess(features1)

            # compare
            diff = np.subtract(features_list, features1)
            dist = np.sum(np.square(diff), axis=1)
            idx = np.argmin(dist)
            if dist[idx] < 1:   # schdule = ?
                predictions = names[idx].split("_")[0]
                print(dist[idx], predictions, idx)

            # draw img
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (80,18,236), 2)
            cv2.rectangle(img_ori, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = predictions
            cv2.putText(img_ori, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)


        cv2.imshow('Video', img_ori)
        if frame_count == 10:
            endtime = time.time()
            print("\rReally fps=%s "%(frame_count/(endtime-start_time)), end="")
            start_time = endtime
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # prepare_features("video_data/")
    # prepare_test_img("video_data/")
    # demo_image()
    demo_video()
