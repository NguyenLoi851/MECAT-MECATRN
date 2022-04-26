# read file

import os

print('.')

mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

print(mecatDataFiles[0])

f = open(mecatDataPath+'/'+mecatDataFiles[0], "r")
print(f.read())


