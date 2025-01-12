import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot
import numpy as np

# RGBD_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize([204, 190]),
#     # transforms.CenterCrop(224),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# image = cv2.imread("../rgbd/rgbd_plant/rgb_04_009_01.png")
# label = cv2.imread("../rgbd/rgbd_label/label_04_009_01.png")
# label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# label = RGBD_transform(label)
# image = RGBD_transform(image)
# label = label.permute(1, 2, 0)
# image = image.permute(1, 2, 0)
# plt.figure(1)
# plt.imshow(label)
# plt.figure(2)
# plt.imshow(image)
# plt.show()
# print(label)

def get_0_1_array(array, rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array

array = np.random.random([2, 3, 3, 3])
re_array = get_0_1_array(array)
dot = array * re_array
print("array", array)
print("re_array", re_array)
print("dot", dot)