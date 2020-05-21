"""
@File : util.py
@Author: Dong Wang
@Date : 2020/5/8
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Circle parameters
radius = 3
circle_color = (0, 0, 255)  # BGR
circle_thickness = 4  #
gestures = ['fist_dorsal', 'fist_palm', 'open_dorsal',
            'open_palm', 'three_fingers_dorsal', 'three_fingers_palm']


def realTimeExtraction():
    trained_model = load_model('./saved_models/Sub1weights-best.hdf5')
    # cap = cv2.VideoCapture('D:/data/IIS/Dataset Subsystem 1/videos/test.webm')
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if not cap.read()[0]:
        cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 180)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("capture")
                cv2.imwrite('./imgs/new.jpg', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
            # frame = Image.fromarray(frame)
            test = np.array([frame / 255])
            print(test.shape)
            pre = trained_model.predict(test)[0]
            pres = pre * 255
            pres[pres < 50] = 0
            num_point = len(pres)
            for i in range(0, num_point, 2):
                x, y = int(pres[i]), int(pres[i + 1])
                if x != 0 and y != 0:
                    cv2.circle(frame, (x, y), radius, circle_color, thickness=circle_thickness)
            cv2.imshow('frame', frame)
    cap.release()


def ExtractionFromImage(filename='./imgs/open_palm_146.jpg'):
    trained_model = load_model('./saved_models/Sub1weights-best.hdf5')

    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
    # frame = Image.fromarray(frame)
    test = np.array([frame / 255])
    print(test.shape)
    pre = trained_model.predict(test)[0]
    pres = pre * 255
    pres[pres < 20] = 0
    num_point = len(pres)
    for i in range(0, num_point, 2):
        x, y = int(pres[i]), int(pres[i + 1])
        if x != 0 and y != 0:
            cv2.circle(frame, (x, y), radius, circle_color, thickness=circle_thickness)

    cv2.imwrite(filename.replace('.jpg', '_result.jpg'), frame)


def realTimeGesture():
    trained_model = load_model('./saved_models/End2endWeights-best.hdf5')
    # cap = cv2.VideoCapture('D:/data/IIS/Dataset Subsystem 1/videos/test.webm')
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if not cap.read()[0]:
        cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 180)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("capture")
                cv2.imwrite('./imgs/new.jpg', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
            # frame = Image.fromarray(frame)
            test = np.array([frame / 255])
            # print(test.shape)
            pre = trained_model.predict(test)[0]
            index = int(np.argmax(pre))
            gesture = gestures[index]
            print(gesture)
            cv2.putText(frame, gesture+' '+str(pre[index]), (5, 30), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
    cap.release()
