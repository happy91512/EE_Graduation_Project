import numpy as np
import cv2
import time
import keras.backend as K
from keras.models import *
from keras.layers import *

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
# HEIGHT=360
# WIDTH=640
sigma = 2.5
mag = 1

# Loss function
def custom_loss(y_true, y_pred):
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
        + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)


class TrackNetV2_33:
    def __init__(self, filename: str) -> None:
        model = load_model(filename, custom_objects={'custom_loss': custom_loss})
        model.summary()
        print('Beginning predicting......')
        self.model = model

    def predict(self, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
        ratio = img1.shape[0] / HEIGHT

        points = np.zeros((3, 2), dtype=np.int16)
        masks = np.zeros((3, *img1.shape[:-1]), dtype=np.uint8)

        # Adjust BGR format (cv2) to RGB format
        x1 = img1[..., ::-1]
        x2 = img2[..., ::-1]
        x3 = img3[..., ::-1]
        # Resize the images
        x1 = cv2.resize(img1, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        x2 = cv2.resize(img2, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        x3 = cv2.resize(img3, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        # Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(x1, -1, 0)
        x2 = np.moveaxis(x2, -1, 0)
        x3 = np.moveaxis(x3, -1, 0)
        # Create data
        unit = np.asarray([[x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2]]])
        unit = unit.astype('float32')
        unit /= 255
        # tt = time.time()
        y_pred = self.model.predict(unit, batch_size=BATCH_SIZE)
        # print('model predict time = ', time.time() - tt)
        y_pred = y_pred[0] > 0.5
        y_pred = y_pred.astype('float32')
        h_pred = y_pred * 255
        h_pred = h_pred.astype('uint8')
        for i in range(3):
            if np.amax(h_pred[i]) <= 0:
                continue

            # h_pred
            (cnts, _) = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            (cx_pred, cy_pred) = (int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2)))

            points[i] = (cx_pred, cy_pred)
            cv2.circle(masks[i], (cx_pred, cy_pred), 5, 255, -1)

        return masks, points
