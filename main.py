# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:22:16 2021

@author: Administrator
"""

from __future__ import print_function

import numpy as np
import os
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from MIL_model import resnext101_32x8d
import GlobalManager as gm
import xlrd
from tqdm import tqdm 

# Training settings
parser = argparse.ArgumentParser(description = 'MIL Leukemia Prediction Example')

parser.add_argument('--data_path', type = str, default = 'D:/data/bone_marrow/training/single_cell_dataset', metavar = 'Path', 
                    help = 'location of single cell dataset')
parser.add_argument('--info_path', type = str, default = 'D:/data/bone_marrow/info.xlsx', metavar = 'Path', 
                    help = 'location of dataset information')
parser.add_argument('--pretrained_model', type = str, default = 'D:/program/AttentionDeepMIL-master/marrow_class_resnext101_20220202.pth', metavar = 'Path', 
                    help = 'pretrained CNN backbone model')
parser.add_argument('--epochs', type = int, default = 30, metavar = 'N',
                    help = 'number of epochs to train (default: 20)')
parser.add_argument('--lr', type = float, default = 1e-5, metavar = 'LR',
                    help = 'learning rate (default: 10e-5)')
parser.add_argument('--step_size', type = int, default = 10,
                    help = 'period of learning rate decay (default: 10)')
parser.add_argument('--gamma', type = float, default = 0.1,
                    help = 'factor of learning rate decay (default: 0.1)')
parser.add_argument('--reg', type = float, default = 10e-5, metavar = 'R',
                    help = 'weight decay')
parser.add_argument('--type_number', type = int, default = 5, metavar = 'T',
                    help = 'bags have a positive labels for leukemia type')
parser.add_argument('--bag_length', type = int, default = 64, metavar = 'ML',
                    help = 'average bag length')
parser.add_argument('--var_bag_length', type = int, default = 0, metavar = 'VL',
                    help = 'variance of bag length')
parser.add_argument('--train_iteration', type = int, default = 1000, metavar = 'NTrain',
                    help = 'repeat sampling for prediction')
parser.add_argument('--testing_repeat_time', type = int, default = 21, metavar = 'NTest',
                    help = 'repeat sampling for prediction')
parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
parser.add_argument('--no-cuda', action = 'store_true', default = False,
                    help = 'disables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')
    
def main(args):    
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    
    data_info = args.info_path
    wb = xlrd.open_workbook(data_info)
    table = wb.sheet_by_name('Sheet1')
    type_info = {}
    for i in range(table.nrows):
        x = table.cell(i,1).value
        y = table.cell(i,0).value
        type_info[y] = x
        
    data_path = args.data_path    
    type_list = os.listdir(os.path.join(data_path,'train'))
    gm._init_()
    gm.set_value("path",data_path)
    from marrowdataloader import trainset,testset
    print('Load Train Set')
    train_loader = data_utils.DataLoader(trainset(),
                                         batch_size = args.bag_length,
                                         shuffle = True,
                                         **loader_kwargs)

    
    print('Init Model')
    
    
    model = resnext101_32x8d(num_classes = 21,pretrained = True, pretrained_model = args.pretrained_model)
    model.fc = torch.nn.Linear(2048, 1024, bias = True)
    if args.cuda:
        model.cuda()
    
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), weight_decay = args.reg)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma = args.gamma, last_epoch = -1)
    train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    type_idx = int(np.random.randint(0,len(type_list),1))
    folder_list = os.listdir(os.path.join(data_path,'train',type_list[type_idx]))
    folder_idx = int(np.random.randint(0,len(folder_list),1))
    train_folder = os.path.join(data_path,'train',type_list[type_idx],folder_list[folder_idx])
    gm.set_value("train_folder",train_folder)
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.
        train_error = 0.
        with tqdm(total = args.train_iteration) as t: 
            count = 0
            for batch_idx, (data, instance_label) in enumerate(train_loader):
                optimizer.zero_grad()
                label = torch.tensor([type_idx])
    
                if args.cuda:
                    data = data.cuda()
                data = Variable(data)
                if args.cuda:
                    data, label = data.cuda(), label.cuda()
    
                data = Variable(data)
                output_sum = model(data)
                output = output_sum[0]
                target = label.long()
                loss = train_loss_fn(output,target)
                train_loss +=  loss.data
                correct = output.argmax().eq(target.view_as(output.argmax())).sum().item()
                train_error += 1-correct
                t.update(1)
                t.set_description('Iteration %i' % count)
                t.set_postfix(loss = loss.item(),error = train_error/(count+1))
                loss.backward()
                optimizer.step()
                count += 1
                if count>args.train_iteration:
                    break
                type_idx = int(np.random.randint(0,len(type_list),1))
                folder_list = os.listdir(os.path.join(data_path,'train',type_list[type_idx]))
                folder_idx = int(np.random.randint(0,len(folder_list),1))
                train_folder = os.path.join(data_path,'train',type_list[type_idx],folder_list[folder_idx])
                gm.set_value("train_folder",train_folder)
                
        scheduler.step()
        t.close()
        train_loss /=  (count+1)
        train_error /=  (count+1)
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, lr: {:.7f}'.format(epoch, train_loss, train_error, optimizer.state_dict()['param_groups'][0]['lr']))
        
        
    print('Load Test Set')
    test_loader = data_utils.DataLoader(testset(),
                                         batch_size = args.bag_length,
                                         shuffle = True,
                                         **loader_kwargs)  
    
    folder_list = os.listdir(os.path.join(data_path,'test'))
    repeat = args.testing_repeat_time
    cm = np.zeros([5,5])
    ouput_record = np.zeros([repeat])-1
    folder_list = os.listdir(os.path.join(data_path,'test'))
    cm = np.zeros([args.type_number,args.type_number])
    acc = 0
    fn = 0
    bag_label = -1
    with torch.no_grad():
        for folder in folder_list:
            bag_label += 1
            test_list = os.listdir(os.path.join(data_path,'test',folder))
            fn += len(test_list)
            for file in test_list:
                test_folder = os.path.join(data_path,'test',folder,file)
                gm.set_value("train_folder",test_folder)
                count = 0
                tp = 0
                for batch_idx, (data, label) in enumerate(test_loader):
                    target = torch.tensor([bag_label]).long()          
                    if args.cuda:
                        data = data.cuda()
                    data = Variable(data)
                    output_sum = model(data)
                    output = output_sum[0]
                    pred = output_sum[2][0].cpu().numpy()[0]
                    ouput_record[count] = pred
                    count += 1
                    ouput_record = np.uint8(ouput_record)    
                    type_counts = np.bincount(ouput_record)
                    pred_res = np.argmax(output.cpu().numpy()[0])
                    if pred_res == target.cpu().numpy()[0]:
                        tp+=1
                    if count>=repeat:
                        break
                if tp/repeat>0.5:
                    acc+=1
                cm[bag_label,np.argmax(type_counts)]+=1
        print('accuracy: '+str(acc/fn))
        torch.save(model, './best_mode.pth') #

if __name__  ==  "__main__":
    results = main(args)
    print("finished!")