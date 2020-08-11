import sys, os
import onnxruntime
import numpy as np
import onnx
import cv2

import facenet
import ultra_light


# 获取测试用例和标签
def get_test_data(root="/data/dataset/LFW"):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw_mtcnn_160_jpg'
    nameLs = []
    nameRs = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3]))) 
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, flags]


def get_max_accuary(flags, results):
    flags = np.array(flags)
    threshold = np.arange(0, 3, 0.01)
    accuary_list = []

    for i in threshold:
        p = np.sum(results[flags == 1] < i)
        n = np.sum(results[flags == -1] >= i)
        accuary = 1.0 * (p + n) / len(results)
        accuary_list.append(accuary)

    max_index = np.argmax(np.array(accuary_list))
    print("threshold:%s  max accuary:%s"%(threshold[max_index], accuary_list[max_index]))
    return threshold[max_index], accuary_list[max_index]


def get_accuary(flags, results, threshold=1.1):
    results = np.array(results)
    flags = np.array(flags)
    p = np.sum(results[flags == 1] < threshold)
    n = np.sum(results[flags == -1] >= threshold)
    accuary = 1.0 * (p + n) / len(results)
    print("threshold:%s  accuary:%s"%(threshold, accuary))
    return 1.0 * (p + n) / len(results)


def onnx_test(model="./models/FaceNet_vggface2_optmized.onnx"):
    sess = onnxruntime.InferenceSession(model)
    input_name = sess.get_inputs()[0].name

    nameLs, nameRs, flags = get_test_data()
    print(len(flags))  # 6000个测试用例

    featureLs = []
    featureRs = []
    results = []

    for index in range(len(nameLs)):

        # print(nameLs[index])
        img_ori = cv2.imread(nameLs[index])
        input_data = facenet.preprocess2(img_ori)
        ret1 = sess.run(None, {input_name: input_data})
        ret1 = np.array(ret1[0])
        ret1 = facenet.postprocess(ret1)
        featureLs.append(ret1)
        sys.exit()

        # print(nameRs[index])
        img_ori2 = cv2.imread(nameRs[index])
        input_data = facenet.preprocess2(img_ori2)
        ret2 = sess.run(None, {input_name: input_data}) 
        ret2 = np.array(ret2[0])
        ret2 = facenet.postprocess(ret2)
        featureRs.append(ret2)
        # dist = np.linalg.norm(ret1 - ret2)

        # 欧式距离
        diff = np.subtract(ret1, ret2)
        dist = np.sum(np.square(diff),1)[0]

        print(index, dist, flags[index])
        results.append(dist)

    results = np.array(results)
    np.save("resluts.npy", results)
    get_max_accuary(flags, results)



if __name__ == '__main__':
    onnx_test()