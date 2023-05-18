

import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from scipy.stats import t

parser = argparse.ArgumentParser(description="Loading data.")
parser.add_argument('--obj', type=str, default="train_loss",
                    help="objective of the plot, the options are:'train_loss', 'train''acc', 'val_loss', 'val_acc'.")
parser.add_argument('--path1', type=str, default="none",
                    help="path of the data folder.")
parser.add_argument('--path2', type=str, default="none",
                    help="path of the data folder.")
parser.add_argument('--path3', type=str, default="none",
                    help="path of the data folder.")
parser.add_argument('--path4', type=str, default="none",
                    help="path of the data folder.")
parser.add_argument('--data_order', type=str, default=' '.join(x for x in ["SGD","SVRG","NNAG","NNAG+SVRG","Epoch","Lower1","Upper1","Lower2","Upper2","Lower3","Upper3","Lower4","Upper4"]),
                    help="Data names in order of the paths.")
parser.add_argument('--epoch', type=str, default=100,
                    help="Numbr of epochs.")
parser.add_argument('--outputfile', type=str, default="data_all",
                    help="Name of the output file")
"""
    Used for loading data calculating confidence intervals and further visualization in latex.
    Note that the paths are to be assigned to the folder containing the .npz files. The .npz files are the outputs of the training.
"""




def load_data(path,obj,nepochs):
    files = glob.glob(path +"/*.npz")           # get all the .npz files
    data1=np.zeros(nepochs)
    counter = 0

    for file in files:                   # iterate over the list of files
        with np.load(file, allow_pickle=True) as data:     # open the file
            data0 = dict(data)
            data1 +=(data0[obj])
            counter += 1
    mu = data1/counter # mean of the training Monte-Carlo runs.
    data1=np.zeros(nepochs)
    for file in files:                   # iterate over the list of files
        with np.load(file, allow_pickle=True) as data:     # open the file
            data0 = dict(data)
            data1 +=((data0[obj]-mu)**2)
    s = np.sqrt(1/(counter-1)*data1) # standard deviation of the training Monte-Carlo runs.
    return mu , s , counter


if __name__ == "__main__":
    args = parser.parse_args()
    path1= args.path1
    path2= args.path2
    path3= args.path3
    path4= args.path4
    obj = args.obj
    nepochs = args.epoch
    
    [data1,s1,counter1] = load_data(path1,obj,nepochs)
    [data2,s2,counter2] = load_data(path2,obj,nepochs)
    [data3,s3,counter3] = load_data(path3,obj,nepochs)
    [data4 ,s4,counter4]= load_data(path4,obj,nepochs)
    
 
    # Degrees of freedom
    dof1 = counter1-1
    dof2 = counter2-1
    dof3 = counter3-1
    dof4 = counter4-1

    
    confidence = 0.68 #  Significance level

    
    t_crit1 = np.abs(t.ppf((1-confidence)/2,dof1))
    t_crit2 = np.abs(t.ppf((1-confidence)/2,dof2))
    t_crit3 = np.abs(t.ppf((1-confidence)/2,dof3))
    t_crit4 = np.abs(t.ppf((1-confidence)/2,dof4))
    [lower1,upper1] = [data1-s1*t_crit1/np.sqrt(counter1), data1+s1*t_crit1/np.sqrt(counter1)]
    [lower2,upper2] = [data2-s2*t_crit2/np.sqrt(counter2), data2+s2*t_crit2/np.sqrt(counter2)]
    [lower3,upper3] = [data3-s3*t_crit3/np.sqrt(counter3), data3+s3*t_crit3/np.sqrt(counter3)]
    [lower4,upper4] = [data4-s4*t_crit4/np.sqrt(counter4), data4+s4*t_crit4/np.sqrt(counter4)]                                                                                 
    data_all = args.data_order
 
    for i in range(nepochs):
        
        data_all0 = ' '.join(str(x) for x in [str(data1[i]),str(data2[i]),str(data3[i]),str(data4[i]),str(i+1),str(lower1[i]),str(upper1[i]),str(lower2[i]),str(upper2[i]),str(lower3[i]),str(upper3[i]),str(lower4[i]),str(upper4[i])])
        data_all = '\n'.join([data_all,data_all0])
    outputfile_name = args.outputfile
    file = open(outputfile_name+'.txt','w')
    file.write(data_all)
    file.close()


