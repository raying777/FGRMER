import torch
from torchvision import transforms
from mydataset import MyData1
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("../../")
#from utils import data_generator
#from mymodel import TCN
#from t_g_model_3and6_no_au_and_fusion import NetworkModel #消融分析 no_au_and_fusion
from t_g_model_3and6 import NetworkModel

import numpy as np
import argparse
import copy
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from myutils import plot_train_and_test_result
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=2000,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels_p', type=int, default=4,
                    help='# of point levels (default: 2)')
parser.add_argument('--levels_s', type=int, default=1,
                    help='# of side levels (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-4 ,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#root = './data/mnist'

#Fold_accuracy = 0

TN_Fold = {}
TN_Fold['samm'] = np.zeros(3, dtype=int)
TN_Fold['smic'] = np.zeros(3, dtype=int)
TN_Fold['casme2'] = np.zeros(3, dtype=int)
TN_Fold['total'] = np.zeros(3, dtype=int)

TP_Fold = {}
TP_Fold['samm'] = np.zeros(3, dtype=int)
TP_Fold['smic'] = np.zeros(3, dtype=int)
TP_Fold['casme2'] = np.zeros(3, dtype=int)
TP_Fold['total'] = np.zeros(3, dtype=int)

FP_Fold = {}
FP_Fold['samm'] = np.zeros(3, dtype=int)
FP_Fold['smic'] = np.zeros(3, dtype=int)
FP_Fold['casme2'] = np.zeros(3, dtype=int)
FP_Fold['total'] = np.zeros(3, dtype=int)

FN_Fold = {}
FN_Fold['samm'] = np.zeros(3, dtype=int)
FN_Fold['smic'] = np.zeros(3, dtype=int)
FN_Fold['casme2'] = np.zeros(3, dtype=int)
FN_Fold['total'] = np.zeros(3, dtype=int)
loss_fun = nn.CrossEntropyLoss()
for ai in range(1, 69):
    trainlosses = []
    testlosses = []
    batch_size = args.batch_size
    n_classes = 3
    input_channels = 1
    seq_length = int(1470 / input_channels)
    epochs = args.epochs
    steps = 0
    
    print(args)
    
    
    train_set = MyData1(mat_root='data/mat/feature_9000x1470_shape.mat', 
                       data_root = 'data/train/train'+str(ai)+'.txt',
                       transform=transforms.ToTensor())
    test_set = MyData1(mat_root='data/mat/feature_9000x1470_shape.mat', 
                       data_root = 'data/test/test'+str(ai)+'.txt',
                       transform=transforms.ToTensor())
    adj = r'data\adj\adj_train'+str(ai)+'.npy'
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle= False)
    #train_loader, test_loader = data_generator(root, batch_size)'''
    '''
    permute = torch.Tensor(np.random.permutation(1470).astype(np.float64)).long()
    channel_sizes_p = [args.nhid] * args.levels_p
    channel_sizes_s = [args.nhid] * args.levels_s
    kernel_size = args.ksize
    model = TCN(input_channels, n_classes, channel_sizes_p, channel_sizes_s, kernel_size=kernel_size, dropout=args.dropout)
    '''
    model = NetworkModel(adj)
    if args.cuda:
        model.cuda()
        #permute = permute.cuda()
    
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_pred = []
    best_target = []
    Epoch_F1_score = 0.0
    FP_Epoch = 0.0
    FN_Epoch = 0.0
    TP_Epoch = 0.0
    TN_Epoch = 0.0
    def train(ep):
        #fp = open('I:/codes/project_code/casme2_process/8_tcn/new/out/result2.txt', 'a+')
        fp.write(('-' * 100)+"\n")
        #global steps
        train_loss = 0
        correct_train = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #data_numpy0 = data.numpy()
            if args.cuda: data, target = data.cuda(), target.cuda()
            '''
            data = data.view(-1, input_channels, seq_length)
           # data_numpy1 = data.numpy()
            if args.permute:
                data = data[:, :, permute]
            #data_numpy2 = data.numpy()
            '''
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            pred_train = output.data.max(1, keepdim=True)[1]
            correct_train += pred_train.eq(target.data.view_as(pred_train)).cpu().sum()
            #loss1 = F.nll_loss(output1, target)
            loss = loss_fun(output, target)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            
            train_loss += loss
            #steps += seq_length
            if batch_idx > 0 and (batch_idx + 1) % len(train_loader) == 0:
                print('Train Epoch: {} \tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)'.format(
                    ep, train_loss.item()/len(train_loader), correct_train, len(train_loader.dataset),
                100. * correct_train / len(train_loader.dataset)))
                
                
                fp.write(('Train Epoch: {} \tLoss: {:.6f}\t'.format(
                    ep, train_loss.item()/len(train_loader)))+"\n")            
                
                
                taloss = train_loss.item()/len(train_loader)
                train_loss = 0
        #fp.close()
    
        return taloss
    def test():
        #fp = open('I:/codes/project_code/casme2_process/8_tcn/new/out/result2.txt', 'a+')
        
        model.eval()
        test_loss = 0
        #test_loss1 = 0
        correct = 0
        
        global best_acc
        global best_model_wts
        global best_pred
        global best_target
        global Epoch_accuracy 
        global Epoch_F1_score
        global FP_Epoch
        global FN_Epoch
        global TP_Epoch 
        global TN_Epoch
        
        with torch.no_grad():
            for data, target in test_loader:
                #data_test0 = data.numpy()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                '''
                data = data.view(-1, input_channels, seq_length)
                #data_test1 = data.numpy()
                if args.permute:
                    data = data[:, :, permute]
                #data_test2 = data.numpy()
                '''
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                #test_loss1 += F.nll_loss(output, target, size_average=False).item()
                #test_loss1 += F.nll_loss(output1, target).item()
                test_loss += loss_fun(output, target).item()
                
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                target_npy = target.cpu().numpy()
                target_list = target_npy.tolist()
                pred_npy = pred.cpu().numpy().reshape(len(target_list))
                pred_list = pred_npy.tolist()
            test_loss /= len(test_loader)
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            print('test label: {}'.format(target_list))
            print('pred label: {}\n'.format(pred_list))
            fp.write(('-' * 10)+"\n")
            fp.write(('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset))))
            
            fp.write('test label: {}'.format(target_list))
            fp.write('pred label: {}\n'.format(pred_list))
            epoch_acc = 100. * correct / len(test_loader.dataset)
            
            matrix = confusion_matrix(target.cpu(), pred.cpu(), labels=[0, 1, 2])
            FP = matrix.sum(axis=0) - np.diag(matrix)
            FN = matrix.sum(axis=1) - np.diag(matrix)
            TP = np.diag(matrix)
            TN = matrix.sum() - (FP + FN + TP)

            f1_s = np.ones([3])
            deno = (2 * TP + FP + FN)
            for f in range(3):
                if deno[f] != 0:
                    f1_s[f] = (2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f])
                else:
                    f1_s[f] = 1

            f1 = np.mean(f1_s)

            

            if f1 >= Epoch_F1_score:
                Epoch_F1_score = f1
                Epoch_accuracy = epoch_acc
                FP_Epoch = FP
                FN_Epoch = FN
                TP_Epoch = TP
                TN_Epoch = TN
                #sample_file = temp_file
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                best_pred = pred_list
                best_target = target_list
            #return test_loss
        #model_best.load_state_dict(best_model_wts)
        return best_model_wts, best_acc, best_pred, best_target,FP_Epoch,FN_Epoch,TP_Epoch,TN_Epoch,test_loss
        #fp.close()
    if __name__ == "__main__":
        since = time.time()
        for epoch in range(1, epochs+1): 
            global fp
            fp = open('out/txt/result'+str(ai)+'.txt', 'a+')        
            trainloss = train(epoch)
            best_model_wts, best_acc, best_pre, best_tar,FP_Epoch,FN_Epoch,TP_Epoch,TN_Epoch,testloss =  test()
            
            trainlosses.append(trainloss)
            testlosses.append(testloss)
            
            if best_acc == 100:
                break
            if epoch % 100 == 0:
                lr /= 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        #plot_train_and_test_result(trainlosses,testlosses)            
        if 1 <=ai <= 24:
            TP_Fold['casme2'] += TP_Epoch
            TN_Fold['casme2'] += TN_Epoch
            FN_Fold['casme2'] += FN_Epoch
            FP_Fold['casme2'] += FP_Epoch
        elif 25 <=ai <= 52:
            TP_Fold['samm'] += TP_Epoch
            TN_Fold['samm'] += TN_Epoch
            FN_Fold['samm'] += FN_Epoch
            FP_Fold['samm'] += FP_Epoch
        elif 53 <=ai <= 68:
            TP_Fold['smic'] += TP_Epoch
            TN_Fold['smic'] += TN_Epoch
            FN_Fold['smic'] += FN_Epoch
            FP_Fold['smic'] += FP_Epoch            

        TP_Fold['total'] += TP_Epoch
        TN_Fold['total'] += TN_Epoch
        FN_Fold['total'] += FN_Epoch
        FP_Fold['total'] += FP_Epoch        
        
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict() , 'out/pth/newbest'+str(ai)+'.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
        fp.write('\nBest test Acc: {:4f}'.format(best_acc))
        fp.close()
        fl = open('out/label/result_label'+str(ai)+'.txt', 'a+')
        fl.write('Best  pred: {}\n'.format(best_pre))
        fl.write('    target: {}'.format(best_tar))
        fl.close()


F1_Score = {}
F1_Score['samm'] = (2 * TP_Fold['samm']) / (2 * TP_Fold['samm'] + FP_Fold['samm'] + FN_Fold['samm'])
F1_Score['smic'] = (2 * TP_Fold['smic']) / (2 * TP_Fold['smic'] + FP_Fold['smic'] + FN_Fold['smic'])
F1_Score['casme2'] = (2 * TP_Fold['casme2']) / (2 * TP_Fold['casme2'] + FP_Fold['casme2'] + FN_Fold['casme2'])
F1_Score['total'] = (2 * TP_Fold['total']) / (2 * TP_Fold['total'] + FP_Fold['total'] + FN_Fold['total'])

Recall_Score = {}
Recall_Score['samm'] = TP_Fold['samm'] / (TP_Fold['samm'] + FN_Fold['samm'])
Recall_Score['smic'] = TP_Fold['smic'] / (TP_Fold['smic'] + FN_Fold['smic'])
Recall_Score['casme2'] = TP_Fold['casme2'] / (TP_Fold['casme2'] + FN_Fold['casme2'])
Recall_Score['total'] = TP_Fold['total'] / (TP_Fold['total'] + FN_Fold['total'])

#Total_accuracy = Fold_accuracy / FOLD

Total_F1_Score = {}
Total_F1_Score['samm'] = np.mean(F1_Score['samm'])
Total_F1_Score['smic'] = np.mean(F1_Score['smic'])
Total_F1_Score['casme2'] = np.mean(F1_Score['casme2'])
Total_F1_Score['total'] = np.mean(F1_Score['total'])

Total_Recall = {}
Total_Recall['samm'] = np.mean(Recall_Score['samm'])
Total_Recall['smic'] = np.mean(Recall_Score['smic'])
Total_Recall['casme2'] = np.mean(Recall_Score['casme2'])
Total_Recall['total'] = np.mean(Recall_Score['total'])

print('\nFold F1 score SAMM: ', Total_F1_Score['samm'])
print('Fold F1 score SMIC: ', Total_F1_Score['smic'])
print('Fold F1 score CASMEII: ', Total_F1_Score['casme2'])
print('Fold F1 score: ', Total_F1_Score['total'])

print('Fold Recall score SAMM: ', Total_Recall['samm'])
print('Fold Recall score SMIC: ', Total_Recall['smic'])
print('Fold Recall score CASMEII: ', Total_Recall['casme2'])
print('Fold Recall score: ', Total_Recall['total'])

jieguo = open('out/uf1_uar.txt', 'a+')
jieguo.write('\nFold F1 score SAMM: {} \n'.format(Total_F1_Score['samm']))
jieguo.write('Fold F1 score SMIC: {} \n'.format(Total_F1_Score['smic']))
jieguo.write('Fold F1 score CASMEII: {} \n'.format(Total_F1_Score['casme2']))
jieguo.write('Fold F1 score: {} \n'.format(Total_F1_Score['total']))

jieguo.write('Fold Recall score SAMM: {} \n'.format(Total_Recall['samm']))
jieguo.write('Fold Recall score SMIC: {} \n'.format(Total_Recall['smic']))
jieguo.write('Fold Recall score CASMEII: {} \n'.format(Total_Recall['casme2']))
jieguo.write('Fold Recall score: {} \n'.format(Total_Recall['total']))
jieguo.close()