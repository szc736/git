import os


# 批量重命名文件夹中的图片文件
class BatchRename():
    def __init__(self):
        self.path = './data/CVPPP2017_LSC_training/training/A3labels'  # 表示需要命名处理的文件夹

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)  # 获取文件夹内所有文件个数
        i = 1  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.png'):
                # 初始的图片的格式为png格式的
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.png')
                try:
                    os.rename(src, dst)
                    i = i + 1
                except:
                    continue


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()