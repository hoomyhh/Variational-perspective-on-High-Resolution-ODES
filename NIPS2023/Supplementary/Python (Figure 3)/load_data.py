import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Loading data.")
parser.add_argument('--obj', type=str, default="train_loss",
                    help="objective of the plot, the options are:'train_loss', 'train''acc', 'val_loss', 'val_acc'.")
parser.add_argument('--path1', type=str, default="none",
                    help="path of the data.")
parser.add_argument('--path2', type=str, default="none",
                    help="path of the data.")
parser.add_argument('--path3', type=str, default="none",
                    help="path of the data.")
parser.add_argument('--data_order', type=str, default=' '.join(x for x in ["SGD","SVRG","NNAG","Epoch"]),
                    help="Data names in order of the paths.")
parser.add_argument('--epoch', type=str, default=100,
                    help="Numbr of epochs.")

"""
    Used for loading data and further visualization in latex.
"""


def load_data(path):
    data = np.load(path, allow_pickle=True)
    data = dict(data)

    return data


if __name__ == "__main__":
    args = parser.parse_args()
    path1= args.path1
    path2= args.path2
    path3= args.path3
    obj = args.obj
    nepochs = args.epoch
    # dictionary containing the 'train_loss', 'train''acc', 'val_loss', 'val_acc'
    data1 = load_data(path1)
    data2 = load_data(path2)
    data3 = load_data(path3)


    data1_obj =  data1[obj]
    data2_obj =  data2[obj]
    data3_obj = data3[obj]
    data_all = args.data_order
    #data_all0 = [data1_obj,data2_obj,data3_obj]
    for i in range(nepochs):
        
        data_all0 = ' '.join(str(x) for x in [str(data1_obj[i]),str(data2_obj[i]),str(data3_obj[i]),str(i+1)])
        data_all = '\n'.join([data_all,data_all0])
    file = open('data_all.txt','w')
    file.write(data_all)
    file.close()
