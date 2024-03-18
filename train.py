import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
import time
from model_evaluate import evaluate
from model_vis import draw_train_process
from MobileNetModel import MobileNet
from LeNetModel import LeNet5
from AlexNetModel import AlexNet
from get_mnist import load_mnist_test, load_mnist_train
import numpy as np
import os


def train(train_loader, test_loader, model, criterion, optimizer, device, num_epochs):
    model = model.to(device)
    print("training on ", device)
    batch_count = 0
    train_loss, train_epoch, train_acc = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for image, label in tqdm(train_loader):
            image = image.reshape(-1, 1, 28, 28)

            image = image.to(device)
            label = label.to(device)
            y_hat = model(image)
            loss = criterion(y_hat, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == label).sum().cpu().item()
            n += label.shape[0]
            batch_count += 1
        train_loss.append(train_l_sum / batch_count)
        train_epoch.append(epoch)
        train_acc.append(train_acc_sum / n)
        test_acc = evaluate(test_loader, model, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    draw_train_process(train_epoch, train_loss, train_acc)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_images, train_labels = load_mnist_train('./MNIST/')
    test_images, test_labels = load_mnist_test('./MNIST/')
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    train_set = Data.TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    test_set = Data.TensorDataset(torch.tensor(test_images), torch.tensor(test_labels))

    train_loader = Data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = Data.DataLoader(test_set, batch_size=64, shuffle=False)

    learning_rate = 0.003
    num_epochs = 20
    # model = MobileNet()
    # model = AlexNet()
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train(train_loader, test_loader, model, criterion, optimizer, device, num_epochs)
    torch.save(model.state_dict(), 'LetNet5.pth')
