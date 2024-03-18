import numpy as np
import cv2

# 创建一个级联分类器用于检测人脸
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# 加载已经训练好的人脸数据
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# 加载标签信息
names = ['person1', 'person2']

# 开启摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 设置摄像头窗口大小
cap.set(4, 480)  # 设置摄像头窗口大小

# 设置字体样式
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while True:
    # 获取摄像头图像帧
    ret, frame = cap.read()

    # 将图像转化为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中所有的人脸对象
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    # 循环检测到的人脸对象
    for (x, y, w, h) in faces:
        # 在图像中标注人脸区域
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 调用训练好的识别模型进行人脸识别
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # 当置信度小于100时输出对应标签信息
        if confidence < 100:
            name = names[id]
            confidence_str = "{0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence_str = "{0}%".format(round(100 - confidence))

        # 在人脸框下方输出人名和置信度
        cv2.putText(frame, name, (x + 5, y + h - 5), font, fontscale, fontcolor)
        cv2.putText(frame, confidence_str, (x + 5, y + h + 30), font, fontscale, fontcolor)

    # 显示带有识别结果的图像
    cv2.imshow('frame', frame)

    # 按下q键退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()




