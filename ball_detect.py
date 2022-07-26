import os
import random
import cv2
from right import RightPost
import numpy as np
from paddlelite.lite import MobileConfig, create_paddle_predictor

model_path = 'football0725.nb'
config = MobileConfig()
config.set_model_from_file(model_path)
predictor = create_paddle_predictor(config)
input_tensor = predictor.get_input(0)
input_tensor1 = predictor.get_input(1)
input_c, input_h, input_w = 3, 128,128
put1 = np.array([0.533,0.4])
put1 = put1.reshape([1,2]).astype('float32')

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225] 


def get_ball_box(image):
    image_data = image.copy()
    image_data = cv2.resize(image_data, (input_h, input_w))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = image_data.transpose((2, 0, 1)) /255.0
    image_data = (image_data - np.array(image_mean).reshape(
        (3, 1, 1))) / np.array(image_std).reshape((3, 1, 1))
    image_data = image_data.reshape([1, input_c, input_h, input_w]).astype('float32')
    input_tensor.from_numpy(image_data)
    input_tensor1.from_numpy(put1)
    predictor.run()
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()
    if output_data.any():
        # print(output_data)
        first_out = output_data[0]
        # print(first_out)
        if first_out[1] > 0.3:
            label, pro, x1,y1, x2,y2 = first_out
            # cv2.rectangle(image, (int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            return abs(int(x1)), abs(int(y1)), abs(int(x2)), abs(int(y2))
    return 0, 0, 0, 0
    
    
    
def get_circle(post, image):
    image = post.color_mask(image)
    image = cv2.Canny(image, 30, 100)
    circles1 = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 110, param1=200, param2=10, minRadius=0,
                                maxRadius=0)
    # print(circles1)
    if circles1 is not None:
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        circle_center = [int(circles[0][0]), int(circles[0][1])]
        radium = int(circles[0][2])
    else:
        circle_center = None
        radium = None
    return circle_center, radium



if __name__ == '__main__':
    

    cap = cv2.VideoCapture('videos/right2.avi')

    rightpost = RightPost()

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        x1,y1,x2,y2 = get_ball_box(frame)
        if x1 or y1 or x2 or y2:
            circle_center, radium = get_circle(rightpost, frame[y1:y2,x1:x2])
            print(circle_center)
        else:
            circle_center, radium = None, None
        cv2.imshow('i',frame)
        cv2.waitKey(1)

