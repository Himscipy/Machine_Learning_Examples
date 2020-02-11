import os
import matplotlib.pyplot as plt
import numpy as np
import struct
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser(description='Write the 8M MNIST files in NPZ format image files')
parser.add_argument('-p', '--path_data', type=str, default='./')
parser.add_argument('-f', '--file_name', type=str, default='mnist60k-')
parser.add_argument('-o', '--out_path', type=str, default='./DATA')
parser.add_argument('-of', '--out_file', type=str, default='Mnist_')
parser.add_argument('-oft', '--FileSaveType', type=str, default='npz')
parser.add_argument('-ntf', '--Num_Tf_files', type=int, default=4)
parser.add_argument('-c', '--check', type=bool, default=False)
    
args = parser.parse_args()

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def PlotImage(data):
    plt.imshow(data)
    plt.show()
    return

def Check_SampleSize(Sample, Label):
    if (Sample.shape)[0] == len(Label):
        val = True
    else:
        val = False
    return val 

def File_writer(images,labels,Num_TFfiles, samp_per_file, start_index, name,args) :

    tmp = (os.path.join( args.out_path,'Data_Rank_'+str(Num_TFfiles) ) )
    if os.path.exists (tmp):
        out_datadir = tmp
    else:
        out_datadir = os.mkdir( tmp )  

    for k in range(Num_TFfiles):
        # if args.file_type = 'TF':
        #     writer = tf.io.TFRecordWriter(filename)
        # else:
        #     pass

        end_index = k * samp_per_file + samp_per_file

        print (start_index, end_index)   
        
        # Create data directory for each rank

        if args.FileSaveType == 'npz':
            filename = os.path.join(out_datadir, name + "_Rank_" +str(k) +'.npz')
            np.savez_compressed(filename,Images=images[start_index:end_index,:,:,:],Labels=labels[start_index:end_index])

        elif args.FileSaveType == 'Tfrecord':
            # Write individual samples in the TF record file
            rows = 28
            cols = 28
            depth = 1
            filename = os.path.join(out_datadir, (name + "_Rank_" +str(k) +'.tfrecords'))
            writer = tf.python_io.TFRecordWriter(filename)
            print('Writing', filename)
            
            if start_index > end_index: # This condition happens when writing Test data since there start_index is not zero
                end_index = end_index + start_index  
                print ('Updated Index: ',start_index,end_index)

            for i in range(start_index,end_index):
                image_raw = images[i].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[i])),
                    'image_raw': _bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
            
            writer.close()
        else:
            raise RuntimeError('Input file type {} not implemented'.format(args.FileSaveType))

            # train_x, test_x, train_y, test_y = images[0:test_index,:,:,:], images[test_index:,:,:,:], labels[0:test_index], labels[test_index:]
        

        start_index = end_index
    return        



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

    Num_TFfiles = args.Num_Tf_files
    name = args.out_file

    test_sample = int(0.2 * N)
    train_sample = N - test_sample # 20 percent of test image remaining training
    
    # Calculate Samples in the File:
    samp_per_file = round(train_sample / Num_TFfiles) 
    samp_per_file_test = round(test_sample / Num_TFfiles) 

    print ("Num Training Examples",train_sample)
    print ("Num Test Examples",test_sample)
    print ("Samples per file Examples",samp_per_file)
    print ("Samples per file Examples",samp_per_file_test)
    
    
    images = np.empty((N, 784), dtype=np.uint8)
    labels = np.empty(N, dtype=np.uint8)


    for i in range(N):
        labels[i] = ord(file.read(1))
        for j in range(784):
            images[i, j] = ord(file_image.read(1))
        
    # Normalize the image 
    # images = images.reshape((N,28,28,1)) / 255.0
    images = images.reshape((N,28,28,1))


    start_index_Train = 0
    start_index_Test = test_sample

    File_writer(images,labels,Num_TFfiles, samp_per_file, start_index_Train, "Train",args)
    File_writer(images,labels,Num_TFfiles, samp_per_file_test, start_index_Test, "Test",args)
    

    # print (images[1,:,:])
    # PlotImage(images[1,:,:])
    

if args.check:
    with np.load(filename) as file:
        out = ( np.array_equal(train_x,file['Train_x']) ) * ( np.array_equal(test_x,file['Test_x']) )*( np.array_equal(train_y,file['Train_y']) ) * ( np.array_equal(test_y,file['Test_y']) )
        out_Len_Train = Check_SampleSize(file['Train_x'],file['Train_y'])
        out_Len_Test = Check_SampleSize(file['Test_x'],file['Test_y'])
        
        out_ = (out_Len_Train * out_Len_Test)

    print ('Check Passed ...!! .npz file successfully written')
    
    if out == 0 or out_ == 0:
        raise RuntimeError('Script Failed...!!')
