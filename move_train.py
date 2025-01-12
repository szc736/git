import os
import shutil
path_img = './data/CVPPP2017_LSC_training/training/A3'
ls = os.listdir(path_img)
for i in ls:
    if i.find('rgb')!=-1: #label是区分的关键词
        shutil.move(path_img+'/'+i, "./data/CVPPP2017_LSC_training/training/A3train/"+i)