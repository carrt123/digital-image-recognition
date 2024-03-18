
import gzip
import numpy as np
import struct


# 定义一个函数，用于从原始文件中加载MNIST数据集
# 需要两个参数：数据存储路径和文件名
def load_mnist_train(path, kind='train'):
    # 设置标签和图片的路径
    labels_path = path + '/' + kind + '-labels-idx1-ubyte.gz'
    images_path = path + '/' + kind + '-images-idx3-ubyte.gz'
    # 打开标签文件
    with gzip.open(labels_path, 'rb') as lbpath:
        # 读取前八个字节，并根据格式化字符串'>II'解包成两个整数
        struct.unpack('>II', lbpath.read(8))
        # 读取剩下的字节，并用np.frombuffer()函数将其转换为numpy数组
        # 要指定数据类型为uint8，即一个8位无符号整数
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    # 打开图片文件
    with gzip.open(images_path, 'rb') as imgpath:
        # 读取前16个字节，并根据格式化字符串'>IIII'解包成四个整数
        struct.unpack(">IIII", imgpath.read(16))
        # 读取剩下的字节，并用np.frombuffer()函数将其转换为numpy数组
        # 要指定数据类型为uint8，即一个8位无符号整数
        # 并将其reshape成一个len(labels)行、784列的二维数组
        # 每一行代表一张图片，784个元素代表这张图片的像素值
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    # 返回numpy数组images和labels
    return images, labels


def load_mnist_test(path, kind='t10k'):
    """从原始文件中加载MNIST数据集
    path: 存储数据的路径"""
    labels_path = path + '/' + kind + '-labels-idx1-ubyte.gz'
    images_path = path + '/' + kind + '-images-idx3-ubyte.gz'
    with gzip.open(labels_path, 'rb') as lbpath:
        # 读取前八个字节，并根据格式化字符串'>II'解包成两个整数
        struct.unpack('>II', lbpath.read(8))
        # 读取剩下的字节，并用np.frombuffer()函数将其转换为numpy数组
        # 要指定数据类型为uint8，即一个8位无符号整数
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    # 打开图片文件
    with gzip.open(images_path, 'rb') as imgpath:
        # 读取前16个字节，并根据格式化字符串'>IIII'解包成四个整数
        struct.unpack(">IIII", imgpath.read(16))
        # 读取剩下的字节，并用np.frombuffer()函数将其转换为numpy数组
        # 要指定数据类型为uint8，即一个8位无符号整数
        # 并将其reshape成一个len(labels)行、784列的二维数组
        # 每一行代表一张图片，784个元素代表这张图片的像素值
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        # 返回numpy数组images和labels
    return images, labels

# 在这个修改后的代码中，我们使用自定义的`load_mnist_train`和`load_mnist_test`函数来加载MNIST数据集，这些函数使用`gzip`模块和`numpy`执行适当的解压和格式化操作。
# 需要注意的是，我们将训练集数据导入为一维784像素数组，然后在之后将其转换为二维28 x 28数组。
# 此外，我们在加载并格式化数据集之后使用`TensorDataset`将预处理的图像和标签组合成数据对，并使用`DataLoader`加载数据集。最后，我们可以将此数据集提供给一个PyTorch模型，以用于训练和测试。
