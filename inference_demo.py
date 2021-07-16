
import numpy as np
import onnxruntime
import onnx
import cv2

import argparse
import json
import time


D_TYPE = "float32"

network_param = {
    "shufflenetv1": {
        "TO_RGB": False,
        "MEAN": [0, 0, 0],
        "STDDEV": [0.00392, 0.00392, 0.00392]
    },
    "efficientnet_lite0": {
        "TO_RGB": True,
        "MEAN": [0.498, 0.498, 0.498],
        "STDDEV": [0.502, 0.502, 0.502]
    }
}


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def preprocess(input_data, size, model_name):
    # convert the input data into the float32 input
    img_data = input_data.astype(D_TYPE)

    # normalize
    mean_vec = np.array(network_param[model_name]['MEAN'])
    stddev_vec = np.array(network_param[model_name]['STDDEV'])

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=1, 
                        help='model name')
    args = parser.parse_args()
    print(args)
    model_name = args.model_name
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
    if network_param[model_name]['TO_RGB']:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))

    image_data = image.transpose(2, 0, 1)
    input_data = preprocess(image_data, size, model_name)

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


