
import numpy as np
import onnxruntime
import onnx
import cv2

import json
import time


D_TYPE = "float32"

MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]



def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def preprocess(input_data, size):
    # convert the input data into the float32 input
    img_data = input_data.astype(D_TYPE)

    # normalize
    mean_vec = np.array(MEAN_RGB)
    stddev_vec = np.array(STDDEV_RGB)

    norm_img_data = np.zeros(img_data.shape).astype(D_TYPE)
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, size, size).astype(D_TYPE)
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


if __name__ == '__main__':
    model_name = "efficientnet_lite0"
    onnx_name = "{}.onnx".format(model_name)
    session = onnxruntime.InferenceSession(onnx_name, None)

    # get the name of the first input of the model
    inputs = session.get_inputs()[0]
    input_name = inputs.name
    input_shape = inputs.shape
    print('Input Name:', input_name)
    print('Input Shape:', input_shape)
    size = input_shape[3]

    labels = load_labels('imagenet-simple-labels.json')
    image = cv2.imread('space_shuttle_299x299.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))

    image_data = image.transpose(2, 0, 1)
    input_data = preprocess(image_data, size)

    start = time.time()
    raw_result = session.run([], {input_name: input_data})
    end = time.time()
    print("inference cost time: {}".format(end - start))

    res = postprocess(raw_result)
    raw_res = raw_result[0].reshape(-1)

    idx = np.argmax(res)
    print("max probability: {}".format(res[idx]))
    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print('============ Top 5 labels are: ============')
    for idx in sort_idx[:5]:
        print(idx, raw_res[idx], labels[idx])


