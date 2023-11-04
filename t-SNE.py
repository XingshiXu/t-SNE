from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from nets.facenet import Facenet as facenet # 自己的网络
from utils.utils import resize_image, preprocess_input  # 自己定义的算法

# https://zhuanlan.zhihu.com/p/113379115
# https://blog.csdn.net/Avery123123/article/details/104907491
# https://blog.csdn.net/cough777/article/details/118888761

def modelbuild(): # 该函数需要根据具体 模型进行设置
    # 基本参数设定
    backbone = "mobilenet"
    input_shape = [160, 160, 3]
    model_path = r"H:\LRFRcode\facenet-pytorch\logs\ep100-loss0.000-val_loss2.942.pth"
    annotation_path = r"H:\LRFRcode\FaceNet-xxs\cls_train.txt"

    # 实例化模型 载入权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = facenet(backbone=backbone, mode="predict").eval()
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net = net.cuda()
    print('{} model loaded.'.format(model_path))

    # 遍历数据（类别与地址）
    with open(annotation_path,"r") as f:
        lines = f.readlines()

    # 构建空列表用于保存embeddedfeture和label
    emb_feture = []
    label_True = []

    # 打开图像

    for i, Line in enumerate(lines):
        if i>999:# 使用时删除
            break

        print(i)
        with torch.no_grad():
            Line_split = Line.split(";")
            imagepath = Line_split[1].split()[0]
            image_1 = Image.open(imagepath)
            label = int(Line_split[0])
            if (label<10): # 可以在此设置不遍历全部类别
            # if True:
                image_1 = resize_image(image_1, [input_shape[1], input_shape[0]])
                photo_1 = torch.from_numpy(
                    np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
                photo_1 = photo_1.cuda()
                output1 = net(photo_1).cpu().numpy()
                emb_feture.append(output1)
                label_True.append(label)
    return label_True, emb_feture

def get_data(label, embeddedfeture ):

    embeddedfeture = np.array(embeddedfeture) # 需要是ndarray类型 (样本数量,维度)
    embeddedfeture = embeddedfeture.squeeze()
    label = np.array(label)   # 需要是ndarray类型 (样本数量,维度)

    n_samples, n_features = embeddedfeture.shape
    return embeddedfeture, label, n_samples, n_features

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]), # 相应修改，可以画圆
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    print('Start Buildingmodel')
    GoundT, emd = modelbuild()
    print('Start Computing t-SNE embedding')

    data, label, n_samples, n_features = get_data(GoundT, emd)

    # 使用t-SNE算法进行数据降维
    tsne = TSNE(n_components=2, init='pca', random_state=0) # 实例化一个TSNE类
    result = tsne.fit_transform(data)

    # 结果显示
    fig = plot_embedding(result, label, "t-SNE visualization")
    plt.show(fig)

if __name__ == '__main__':
    main()