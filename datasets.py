import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])

label_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])

def get_CVPPP_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path:
    :return: images_list, labels_list'''

    image_list = []
    label_list = []
    dir_path = Path(dir_path)
    image_path = dir_path / 'A4train'
    label_path = dir_path / "A4labels"
    for img_path in image_path.glob('*.png'):
        image_list.append(str(img_path))
    for label_path in label_path.glob('*.png'):
        label_list.append(str(label_path))
    return image_list, label_list

class CVPPPDatasets(Dataset):
    def __init__(self, dir_path='./data/CVPPP2017_LSC_training/training', image_transform=image_transform, lanel_transform=label_transform):
        self.dir_path = dir_path  # 数据集根目录
        self.img_transform = image_transform
        self.label_transform = label_transform
        self.images, self.labels = get_CVPPP_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_transform:
            img = self.img_transform(img)
            label = self.label_transform(label)
        return img, label


def get_train_valid_sampler(trainset, valid=0.2):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

# dataset = CVPPPDatasets(dir_path="./data/CVPPP2017_LSC_training/training")
# train_sampler, valid_sampler = get_train_valid_sampler(dataset, 0.1)
# dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=train_sampler)
# for image, label in dataloader:
#     label = label.squeeze().permute(1, 2, 0)
#     image = image.squeeze().permute(1, 2, 0)
#     plt.figure(1)
#     plt.imshow(label)
#     plt.figure(2)
#     plt.imshow(image)
#     plt.show()
