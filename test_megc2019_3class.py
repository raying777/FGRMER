# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:22:32 2020

@author: ray
修改记录：46，49 行是版本更新后的警告，改过之后就好了。
"""

import torch
from mydataset import MyData1
from torchvision import transforms
from torch.autograd import Variable
#import torch.nn.functional as F
from t_g_model_3and6 import NetworkModel
from sklearn.metrics import accuracy_score, recall_score, f1_score

print('the test in loso(68 in total) of megc 2019(3 class) :')
batch_size = 64
#input_channels = 1 
#seq_length = int(1470 / input_channels)
#n_classes = 3
#total_correct = []
#total_test_number = []
#total_acc2 = 0

all_target_list = []
all_pred_list = []  
for index in range(1,69):
    
    test_set = MyData1(mat_root=r'.\data\mat\feature_9000x1470_shape.mat', 
                           data_root = r'.\data\test\test'+str(index)+'.txt',
                           transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    adj = r'data\adj\adj_train'+str(index)+'.npy'
    #读数据
   
    model = NetworkModel(adj)
    
    net = torch.load(r'.\data\model\newbest'+str(index)+'.pth')
    model.load_state_dict(net)
    #print(net)
    model.cuda()
    model.eval()
    #test_loss = 0
    #correct = 0
    #all_target_list = []
    #all_pred_list = []        
#    if index == 0: 
#        print(model)
    
    for data, target in test_loader:
    
        data, target = data.cuda(), target.cuda()
        #data = data.view(-1, input_channels, seq_length)
        #data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)
        output = model(data)
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        #test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        target_npy = target.cpu().numpy()
        target_list = target_npy.tolist()
        all_target_list.extend(target_list)
        
        pred_npy = pred.cpu().numpy().reshape(len(target_list))
        pred_list = pred_npy.tolist()
        all_pred_list.extend(pred_list)
        
        fold_uf1 = f1_score(target_list, pred_list, average='macro')
        fold_uar = recall_score(target_list, pred_list, average='macro')
        #total_correct.append(int(correct))
    #test_loss /= len(test_loader.dataset)
    #total_test_number.append(len(test_loader.dataset))
    print('the {}th fold:  test uf1:{:.2%}  test uf1:{:.2%} '.format(index, fold_uf1, fold_uar))    

#total_test_samples = sum(total_test_number)
#total_correct_samples = sum(total_correct)
#total_acc =  total_correct_samples/total_test_samples



camse2_tar_list = all_target_list[0:145]
samm_tar_list = all_target_list[145:278]
smic_tar_list = all_target_list[278:442]

camse2_pre_list = all_pred_list[0:145]
samm_pre_list = all_pred_list[145:278]
smic_pre_list = all_pred_list[278:442]

acc = accuracy_score(all_target_list, all_pred_list)
uar = recall_score(all_target_list, all_pred_list,average='macro')
uf1 = f1_score(all_target_list, all_pred_list,average='macro')
#print('total acc1: {}'.format(total_acc))
print('total acc: {:.2%}'.format(acc))

print('total uf1: {:.2%}'.format(uf1))
print('total uar: {:.2%}'.format(uar))


casme2_uar = recall_score(camse2_tar_list, camse2_pre_list,average='macro')
casme2_uf1 = f1_score(camse2_tar_list, camse2_pre_list,average='macro')

samm_uar = recall_score(samm_tar_list, samm_pre_list,average='macro')
samm_uf1 = f1_score(samm_tar_list, samm_pre_list,average='macro')

smic_uar = recall_score(smic_tar_list, smic_pre_list,average='macro')
smic_uf1 = f1_score(smic_tar_list, smic_pre_list,average='macro')

print('casme2 uf1: {:.2%}'.format(casme2_uf1))
print('casme2 uar: {:.2%}'.format(casme2_uar))

print('samm uf1: {:.2%}'.format(samm_uf1))
print('samm uar: {:.2%}'.format(samm_uar))

print('smic uf1: {:.2%}'.format(smic_uf1))
print('smic uar: {:.2%}'.format(smic_uar))



'''
#another acc 
for i in range(0,26):
    total_acc1 = total_correct[i]/total_test_number[i]
    total_acc2+= total_acc1
total_acc2 = total_acc2 / 26 

print('total acc: {}'.format(total_acc))
print('another total acc: {}'.format(total_acc2))
'''