import tkinter as tk
from PIL import ImageGrab
from MobileNetModel import MobileNet
from LeNetModel import LeNet5
import torch
import numpy as np
from tkinter import messagebox


class PaintMnistAPP:
    def __init__(self, name):
        self.pred = None
        self.prob = None
        self.root = name
        self.root.title("Mnist手写数字识别")
        self.root.geometry('500x400')
        self.root.resizable(width=False, height=False)
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='black')
        self.canvas.place(x=220, y=0)
        self.canvas.bind('<B1-Motion>', self.draw)

        self.save_button = tk.Button(self.root, text="Save", command=self.save_canvas)
        self.save_button.place(width=100, height=50, x=10, y=10)
        self.clean_button = tk.Button(self.root, text="Clean", command=self.clean_cv)
        self.clean_button.place(width=100, height=50, x=10, y=60)
        self.recognize_button = tk.Button(self.root, text="Recognize", command=self.recognize)
        self.recognize_button.place(width=100, height=50, x=10, y=110)

        self.color = 'white'
        self.lineWidth = 3
        self.last_x, self.last_y = None, None

        self.model = None
        self.image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=self.lineWidth, fill=self.color,
                                    capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = event.x, event.y

    def save_canvas(self):
        # 获取画板相对于屏幕的坐标，计算出画板左上角和右下角的绝对坐标
        root_x = self.root.winfo_rootx() + self.canvas.winfo_x()
        root_y = self.root.winfo_rooty() + self.canvas.winfo_y()
        canvas_x = root_x + self.canvas.winfo_width()
        canvas_y = root_y + self.canvas.winfo_height()

        # 按照坐标对画板进行截图，并转化为灰度图并进行 resize
        image_grab = ImageGrab.grab().crop((root_x, root_y, canvas_x, canvas_y)).convert('L').resize((28, 28))
        image_array = np.array(image_grab, dtype=np.float32) / 255.0
        return torch.tensor(image_array.reshape(-1, 1, 28, 28))

    def load_model(self):
        # 创建模型实例，并加载预训练权重文件
        # self.model = MobileNet()
        # self.model.load_state_dict(torch.load('MobileNet.pth', map_location=self.device))
        self.model = LeNet5()
        self.model.load_state_dict(torch.load('LetNet5.pth'))
        # 将模型移动到可用设备上（CPU 或 GPU）
        self.model.to(self.device)
        # 将模型设置为评估模式
        self.model.eval()

    def recognize(self):
        # 对图像数据进行预处理
        x = self.save_canvas()

        # 利用预处理后的数据进行预测，得到预测结果以及概率向量
        with torch.no_grad():
            output = self.model(x.to(self.device))
            self.prob = torch.softmax(output, dim=1)
            self.pred = torch.argmax(self.prob, dim=1)
        print(self.prob)
        pred_num = int(self.pred.cpu().numpy())
        conf = float(self.prob[0][pred_num])
        print(pred_num)
        print(conf)

    def clean_cv(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.last_x, self.last_y = None, None


if __name__ == '__main__':
    root = tk.Tk()
    app = PaintMnistAPP(root)
    root.mainloop()
