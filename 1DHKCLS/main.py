import os
import argparse
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
#import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
from func import load,product,count_parameters_in_MB
from network import hk1dcls_srh, hk1dcls_evl, operate
import time
import logging
import sys

####################################load dataset(indian_pines & pavia_univ & ksc)######################

a=load()

parser = argparse.ArgumentParser(description="1dhkcls")
parser.add_argument('--flag', type=str, default='indian', help='dataset',choices=['pavia','houston','ksc','salina','longkou','hanchuan','honghu','indian'])
parser.add_argument('--block_num',type=int, default=None, help='number of blocks')
parser.add_argument('--layer_num',type=int, default=None, help='number of layers')
parser.add_argument('--exp_num',type=int, default=None, help='number of experiments')
args = parser.parse_args()

All_data,labeled_data,rows_num,categories,r,c,flag=a.load_data(flag=args.flag)

print('Data has been loaded successfully!')

#################################### 归一化 ######################
#设置归一化范围
mi=0
ma=1
a=product(flag, 0)
Alldata_norm=a.normlization(All_data[:,1:-1],mi,ma)

#################################### 空间数据，训练、检验、测试、预测 ###########################

#生成训练样本
Experiment_result=np.zeros([categories+5,12])#OA,AA,kappa，重复10次实验

#实验次数
Experiment_num=args.exp_num

path = 'save'

if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(os.path.join(path,str(flag))):
    os.mkdir(os.path.join(path,str(flag)))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./save/'+str(flag)+'/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#block_layer_acc_matrix=np.zeros([8,10])
#
# for ii in range(3,11):
#     for jj in range(1,11):
#         args.block_num = ii
#         args.layer_num = jj

print('block_num',args.block_num,'layer_num',args.layer_num)

best_OA = 0

for count in range(0,Experiment_num):

    a=product(flag, count)
    rows_num,trn_num,val_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num,All_data)

    print('Experiment {}, dataset preparation Finished!'.format(count))

    ################################### numpy2tensor ####################################

    #label
    y_trn=All_data[trn_num,-1]
    y_val=All_data[val_num,-1]

    print('Experiment {}, Label preparation Finished!'.format(count))

    trn_XX=torch.from_numpy(Alldata_norm[trn_num,:])
    val_XX=torch.from_numpy(Alldata_norm[val_num,:])


    trn_YY=torch.from_numpy(y_trn-1)#标记从0开始
    val_YY=torch.from_numpy(y_val-1)

    print('Experiment {}, Tensor preparation Finished!'.format(count))

    #################################### Searching #####################################

    torch.cuda.empty_cache()#GPU memory released

    trn_dataset=TensorDataset(trn_XX,trn_YY)
    trn_loader=DataLoader(trn_dataset, batch_size=48, sampler=SubsetRandomSampler(range(trn_XX.shape[0])))

    val_dataset = TensorDataset(val_XX, val_YY)
    val_loader = DataLoader(val_dataset, batch_size=48)

    #config lr & epoch

    srh_lr = 1e-2
    srh_epoch = 600

    net = hk1dcls_srh(trn_XX.shape[-1], categories-1, args.block_num, args.layer_num, init_weights=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    logging.info("Srh param size = %fMB", count_parameters_in_MB(net))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=srh_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(srh_epoch))
    loss_srh = []

    srh_time1 = time.time()

    a = operate()

    best_oa = 0

    for i in range(1, srh_epoch+1):
        loss_srh = a.train(i, loss_srh, net, optimizer, scheduler, trn_loader, criterion)

        if i % 50==0:
            y_pred_val = a.inference(net, val_loader, criterion, MODE='VAL')

            oa = np.mean(y_val == y_pred_val)
            kappa = cohen_kappa_score(y_val, y_pred_val)

            logging.info('Experiment {}, srh_epoch {}, valid OA={}'.format(count, i, oa))
            logging.info('Experiment {}, srh_epoch {}, valid Kappa={}'.format(count, i, kappa))


    srh_time2 = time.time()

    loss_srh = np.array(loss_srh)

    network_arch = np.array(net.network_structure())

    #print(type(loss_trn))  ######CPU
    # plt.figure(1)
    # plt.plot(np.array(loss_trn), label='Training')
    # plt.legend()
    # plt.show()

    ##save training model

    logging.info('Experiment {}, genotype'.format(count))
    logging.info(str(network_arch))
    logging.info('Experiment {}, searching finished!'.format(count))

    ######################################### training ####################################

    torch.cuda.empty_cache()  # GPU memory released

    trn_dataset = TensorDataset(trn_XX, trn_YY)
    trn_loader = DataLoader(trn_dataset, batch_size=48, sampler=SubsetRandomSampler(range(trn_XX.shape[0])))

    val_dataset = TensorDataset(val_XX, val_YY)
    val_loader = DataLoader(val_dataset, batch_size=48)

    # config lr & epoch

    trn_lr = 1e-2
    trn_epoch = 1000

    net = hk1dcls_evl(trn_XX.shape[-1], categories - 1, args.block_num, args.layer_num, network_arch, init_weights=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    logging.info("Trn param size = %fMB", count_parameters_in_MB(net))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=trn_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(trn_epoch))
    loss_trn = []

    trn_time1 = time.time()

    a = operate()

    best_oa = 0

    for i in range(1, trn_epoch + 1):
        loss_trn = a.train(i, loss_trn, net, optimizer, scheduler, trn_loader, criterion)

        if i % 50 == 0:
            y_pred_val = a.inference(net, val_loader, criterion, MODE='VAL')

            oa = np.mean(y_val == y_pred_val)
            kappa = cohen_kappa_score(y_val, y_pred_val)

            logging.info('Experiment {}, trn_epoch {}, valid OA={}'.format(count, i, oa))
            logging.info('Experiment {}, trn_epoch {}, valid Kappa={}'.format(count, i, kappa))

    trn_time2 = time.time()

    loss_trn = np.array(loss_trn)

    # print(type(loss_trn))  ######CPU
    # plt.figure(1)
    # plt.plot(np.array(loss_trn), label='Training')
    # plt.legend()
    # plt.show()

    ##save training model

    print('Experiment {}, training finished!'.format(count))

    ######################################### inference ####################################

    y_tes = All_data[tes_num, -1]

    tes_XX = torch.from_numpy(Alldata_norm[tes_num, :])
    tes_YY = torch.from_numpy(y_tes - 1)

    ################### 推断，测试集 ################

    tes_dataset=TensorDataset(tes_XX,tes_YY)
    tes_loader=DataLoader(tes_dataset,batch_size=500)

    a=operate()

    tes_time1 = time.time()
    y_pred_tes=a.inference(net,tes_loader,criterion, MODE='TEST')
    tes_time2 = time.time()

    ####################################### Assess, 测试集 ###########################################

    logging.info('==================Test set=====================')
    logging.info('Experiment {}, test OA={}'.format(count,np.mean(y_tes==y_pred_tes)))
    logging.info('Experiment {}, test Kappa={}'.format(count,cohen_kappa_score(y_tes,y_pred_tes)))

    OA = np.mean(y_tes==y_pred_tes)
    Kappa = cohen_kappa_score(y_tes,y_pred_tes)


    best_arch = 0

    if OA > best_OA:
        best_OA = OA
        best_arch = network_arch
        torch.save(net.state_dict(), './save/{}/hk1dcls_{}.pth'.format(flag, flag))
        np.save('./save/{}/trn_num.npy'.format(flag), trn_num)
        np.save('./save/{}/pre_num.npy'.format(flag), pre_num)
        np.save('./save/{}/best_arch.npy'.format(flag), best_arch)

    ########## 各类别精度

    num_tes=np.zeros([categories-1])
    num_tes_pred=np.zeros([categories-1])

    y_tes = np.array(y_tes).astype(int)

    for k in y_tes:
        num_tes[k-1]=num_tes[k-1]+1
    for j in range(y_tes.shape[0]):
        if y_tes[j]==y_pred_tes[j]:
            num_tes_pred[y_tes[j]-1]=num_tes_pred[y_tes[j]-1]+1

    Acc=num_tes_pred/num_tes*100

    Experiment_result[0,count]=OA*100#OA
    Experiment_result[1,count]=np.mean(Acc)#AA
    Experiment_result[2,count]=Kappa*100#Kappa
    Experiment_result[3, count] = srh_time2 - srh_time1
    Experiment_result[4, count] = trn_time2 - trn_time1
    Experiment_result[5, count] = tes_time2 - tes_time1
    Experiment_result[6:,count] = Acc

    logging.info('Experiment {}, genotype:'.format(count))
    logging.info(str(network_arch))
    logging.info('Experiment {}, Testing set OA={:.4f}'.format(count, OA * 100))
    logging.info('Experiment {}, Testing set Kappa={:.4f}'.format(count, Kappa * 100))
    for i in range(categories-1):
        print('Experiment {} Class {}: {:.4f}'.format(count, i + 1, Acc[i]))

    print('Experiment {}, evaluation finished'.format(count))

scio.savemat('./save/'+str(flag)+'/hk1dcls_loss_srh_'+str(flag)+'.mat',{'hk1dcls_loss_srh':loss_srh})
scio.savemat('./save/'+str(flag)+'/hk1dcls_loss_trn_'+str(flag)+'.mat',{'hk1dcls_loss_trn':loss_trn})

# block_layer_acc_matrix[ii-3,jj-1]= OA * 100
# scio.savemat('./save/'+str(flag)+'/hk1dcls_blk_layer_acc_'+str(flag)+'.mat',{'data':block_layer_acc_matrix})

########## 计算多次实验的均值与标准差并保存

Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

logging.info('BEST genotype:')
logging.info(str(best_arch))

scio.savemat('./save/'+str(flag)+'/hk1dcls_result_'+str(flag)+'.mat',{'data':Experiment_result})

####################################### Classification map #######################################

trn_num = np.load('./save/{}/trn_num.npy'.format(flag))
pre_num = np.load('./save/{}/pre_num.npy'.format(flag))
best_arch = np.load('./save/{}/best_arch.npy'.format(flag))

y_trn = All_data[trn_num, -1]
y_pre = All_data[pre_num, -1]  # include background

pre_XX=torch.from_numpy(Alldata_norm[pre_num,:])
pre_YY=torch.from_numpy(np.ones([y_pre.shape[0]]))

######### 推断，预测集 #########

pre_dataset=TensorDataset(pre_XX,pre_YY)
pre_loader=DataLoader(pre_dataset, batch_size=500)

net = hk1dcls_evl(pre_XX.shape[-1], categories - 1, args.block_num, args.layer_num, best_arch, init_weights=True)

net.load_state_dict(torch.load('./save/{}/hk1dcls_{}.pth'.format(flag, flag), map_location='cpu'))
net=net.cuda()

a=operate()
y_pred_pre=a.inference(net,pre_loader,criterion, MODE='PRED')

print('prediction finished！')

#############  展示 ###########

y_disp=np.zeros([All_data.shape[0]])

y_disp_all=y_disp.copy()
y_disp_all[trn_num]=y_trn
y_disp_all[pre_num]=y_pred_pre

#####################  保存预测全图 #####################

#np.save('./save/'+str(flag)+'/hd1dcls_all_'+str(flag)+'.npy', y_disp_all.reshape(r,c)-1)



def colormap(dataset, x):

    if 'pavia' in dataset:
        y_cmp_dict=scio.loadmat(r'../../../Dataset/whu_hi/color_9.mat')
        y_cmp=y_cmp_dict['Colors_9']
    elif 'hanchuan' in dataset:
        y_cmp_dict=scio.loadmat(r'../../../Dataset/whu_hi/color_16.mat')
        y_cmp=y_cmp_dict['Colors_16']
    elif 'honghu' in dataset:
        y_cmp_dict=scio.loadmat('../../../Dataset/whu_hi/color_22.mat')
        y_cmp=y_cmp_dict['Colors_22']
    
    x = x.astype('uint8')
    #y_cmp[0,:] = np.array([1.0, 1.0, 1.0])
    data_disp=np.zeros([x.shape[0],x.shape[1],3],dtype='uint8')

    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
                data_disp[ii,jj,:]=y_cmp[x[ii,jj],:]*255
    
    return data_disp
    
    
from PIL import Image

pred = y_disp_all.reshape(r,c)
pred = colormap(flag, pred)

pred = Image.fromarray(pred)
pred.save('./save/'+str(flag)+'/hk1dcls_all_'+str(flag)+'.png')

print('Classification map saving finished')