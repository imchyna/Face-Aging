import os
import re
path_dir = os.path.abspath('../DataSet/CASIA-WebFace')
def GetFileList(dir):
    newDir = dir
    if os.path.isfile(dir):
        # fileList.append(dir.decode('gbk'))
        filename = dir.split('/')[6]+'_'+dir.split('/')[7]
        img = imread(dir)
        imsave('../DataSet/CASIA/'+filename,img)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            GetFileList(newDir)


list = GetFileList(path_dir)
