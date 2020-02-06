import os
import matplotlib.pyplot as plt
import numpy as np
import struct
import argparse

parser = argparse.ArgumentParser(description='Write the 8M MNIST files in NPZ format image files')
parser.add_argument('-p', '--path_data', type=str, default='./')
parser.add_argument('-f', '--file_name', type=str, default='mnist60k-')
parser.add_argument('-o', '--out_path', type=str, default='./DATA')
parser.add_argument('-c', '--check', type=bool, default=False)
    
args = parser.parse_args()


def Check_SampleSize(Sample, Label):
    if (Sample.shape)[0] == len(Label):
        val = True
    else:
        val = False
    return val 


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

    test_index = N - int(0.2 *N) # 20 percent of the
    
    print (test_index)
    
    for i in range(N):
        labels[i] = ord(file.read(1))
        for j in range(784):
            images[i, j] = ord(file_image.read(1))

    # Reshaping array with 1 channel form
    images = images.reshape((N,28,28,1))

train_x, test_x, train_y, test_y = images[0:test_index,:,:,:], images[test_index:,:,:,:], labels[0:test_index], labels[test_index:]


filename= os.path.join(args.out_path, 'Training_images.npz')
np.savez_compressed(filename, 
                    Train_x=train_x, 
                    Test_x=test_x, 
                    Train_y=train_y,
                    Test_y=test_y)

if args.check:
    with np.load(filename) as file:
        out = ( np.array_equal(train_x,file['Train_x']) ) * ( np.array_equal(test_x,file['Test_x']) )*( np.array_equal(train_y,file['Train_y']) ) * ( np.array_equal(test_y,file['Test_y']) )
        out_Len_Train = Check_SampleSize(file['Train_x'],file['Train_y'])
        out_Len_Test = Check_SampleSize(file['Test_x'],file['Test_y'])
        
        out_ = (out_Len_Train * out_Len_Test)

    print ('Check Passed ...!! .npz file successfully written')
    
    if out == 0 or out_ == 0:
        raise RuntimeError('Script Failed...!!')
