import sys, os 
import numpy as np
import cv2
# 根据环境进行适配
sys.path.append("../yolo_test/")
sys.path.append("../yolo_test/erpc_nbdla/lib/erpc/erpc_python")
from nbdla_inference import nbdla_client

import LFW_test
import facenet



def nbdla_tess():

    sess = nbdla_client()
    sess.load_model("facenet", output_scale=0.02812)

    nameLs, nameRs, flags = LFW_test.get_test_data()
    print(len(flags))  # 6000个测试用例

    featureLs = []
    featureRs = []
    results = []

    for index in range(len(nameLs)):

        # print(nameLs[index])
        img_ori = cv2.imread(nameLs[index])
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB) #转为BGR方式  # std 和 mean 在nbdla处理
        cv2.imwrite("./temp.jpg", img)
        ret1 = sess.img_inference("./temp.jpg")
        ret1 =np.expand_dims(ret1, 0)
        ret1 = facenet.postprocess(ret1)
        featureLs.append(ret1)

        # print(nameRs[index])
        img_ori = cv2.imread(nameRs[index])
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB) #转为BGR方式  # std 和 mean 在nbdla处理
        cv2.imwrite("./temp.jpg", img)
        ret2 = sess.img_inference("./temp.jpg")
        ret2 =np.expand_dims(ret2, 0)
        ret2 = facenet.postprocess(ret2)
        featureLs.append(ret2)

        # dist = np.linalg.norm(ret1 - ret2)

        # 欧式距离
        diff = np.subtract(ret1, ret2)
        dist = np.sum(np.square(diff))

        print(index, dist, flags[index])
        results.append(dist)


    results = np.array(results)
    np.save("resluts.npy", results)
    LFW_test.get_max_accuary(flags, results)


if __name__ == '__main__':
    nbdla_tess()