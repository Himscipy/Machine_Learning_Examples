import os
import matplotlib.pyplot as plt
import numpy as np
import struct
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Write the 8M MNIST files in NPZ format image files')
parser.add_argument('-p', '--path_data', type=str, default='./')
parser.add_argument('-f', '--file_name', type=str, default='mnist60k-')
parser.add_argument('-o', '--out_path', type=str, default='./DATA')
parser.add_argument('-c', '--check', type=bool, default=True)

args = parser.parse_args()



LOC = {'loc' : os.path.join(args.path_data,(args.file_name+'labels')),
       'loc_image' : os.path.join(args.path_data,(args.file_name+'pattern'))
      }


# Label Loading
with open(LOC['loc'],'rb') as file, open(LOC['loc_image'], 'rb') as file_image:
    file.read(4)
    file_image.read(4)
    N, = struct.unpack('>i', file.read(4))
    if N != struct.unpack('>i', file_image.read(4))[0]:
        raise RuntimeError('wrong pair of MNIST images and labels')

    file_image.read(8)
    images = np.empty((N, 784), dtype=np.uint8)
    labels = np.empty(N, dtype=np.uint8)

    for i in range(N):
        labels[i] = ord(file.read(1))
        for j in range(784):
            images[i, j] = ord(file_image.read(1))

    # Reshaping array with 1 channel form
    images = images.reshape((N,28,28,1))

train_x, test_x, train_y, test_y = train_test_split(images,labels,test_size=0.2)


filename= os.path.join(args.out_path, 'Training_images.npz')
np.savez_compressed(filename, Train_x=train_x, Test_x=test_x, Train_y=train_y,Test_y=test_y)

if args.check:
    with np.load(filename) as file:
        out = ( np.array_equal(train_x,file['Train_x']) ) *  ( np.array_equal(test_x,file['Test_x']) )*         ( np.array_equal(train_y,file['Train_y']) ) * ( np.array_equal(test_y,file['Test_y']) )
    print ('Check Passed ...!! .npz file successfully written')

    if out == 0:
        raise RuntimeError('Script Failed...!!')
