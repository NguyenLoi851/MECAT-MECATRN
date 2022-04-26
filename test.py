# read file

import os

print('.')

mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

# print(mecatDataFiles[0])

# f = open(mecatDataPath+'/'+mecatDataFiles[0], "r")
# print(f.readline())
# print(f.readline())
# print(f.readline())
# str = f.readline().split()[-1]
# print(str)
# for filename in mecatDataFiles:
#     file = open(mecatDataPath+'/'+filename,"r")
#     print(filename)
#     print(file.readline())
#     print(file.readline())
#     print()

mecatRnDataPath = os.getcwd()+'/dataset4mecat/mecat_rn'

mecatRnDataFiles = os.listdir(mecatRnDataPath)

# for filename in mecatRnDataFiles:
#     file = open(mecatRnDataPath+'/'+filename,"r")
#     print(filename)
#     print(file.readline())
#     print(file.readline())
#     print()

# myDict = dict()
# print(type(myDict))
import numpy as np
a = np.array([3,4,5])
print(a)

a = np.concatenate(a, [2,4,7])
print(a)
