# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:21:39 2020

@author: lei
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_train_and_test_result(train_accs,test_accs):
    '''
    训练测试精度可视化
    '''
    epoches = np.arange(1,len(train_accs)+1,dtype=np.int32)
    plt.plot(epoches,train_accs,label='train loss')
    plt.plot(epoches,test_accs, label='test loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend()
    plt.show()