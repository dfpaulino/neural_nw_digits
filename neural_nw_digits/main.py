import os.path

import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax

from neural_network import Network
import Mnist_Data_reader as mnist_loader
def main():

    proj_dir ='/home/dpaulino/workplace/github2/neural_nw_digits'
    img_dir = proj_dir+'/neural_nw_digits/resources'
    train_img_file_path = os.path.join(img_dir,'train-images-idx3-ubyte/train-images-idx3-ubyte')
    train_lbl_file_path= os.path.join(img_dir,'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_img_file_path= os.path.join(img_dir,'t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_lbl_file_path= os.path.join(img_dir,'t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataset=mnist_loader.MnistDataloader(train_img_file_path,train_lbl_file_path,test_img_file_path,test_lbl_file_path)

    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    print("hello [{}]".format(type(x_train[0])))
    print("shape [{}]".format(np.array(x_train).shape))

    plt.imshow(np.reshape(x_train[5],(28,28)),cmap='gray')
    plt.show()
    print("number is [{}]".format(argmax(y_train[5])))

    training_data =[ (x,y) for x,y in zip(x_train, y_train)]
    test_data = [(x, y) for x, y in zip(x_test, y_test)]

    net = Network([784,30,20,10])
    net.SGD(training_data,30,10,5.0,test_data)

if __name__ == "__main__":
    main()