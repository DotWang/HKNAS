import os
import argparse
import time
import numpy as np
#import matplotlib
# import matplotlib as mpl # use slurm
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
from func import load,product,count_parameters_in_MB
#from network_dwconv_parallel import hk3dseg_srh, hk3dseg_evl, operate
#from network_dwconv_spec_spat import hk3dseg_srh, hk3dseg_evl, operate
#from network_dwconv_spat_spec import hk3dseg_srh, hk3dseg_evl, operate
from network_3dconv import hk3dseg_srh, hk3dseg_evl, operate
import logging
import sys
from torch.autograd import Variable

############ parameters setting ############

parser = argparse.ArgumentParser(description="3dhkcls")
parser.add_argument('--flag', type=str, default='indian', help='dataset',
                    choices=['pavia', 'houston', 'ksc', 'salina', 'longkou', 'hanchuan', 'honghu', 'indian'])
parser.add_argument('--block_num', type=int, default=None, help='number of blocks')
parser.add_argument('--layer_num', type=int, default=None, help='number of layers')
parser.add_argument('--exp_num', type=int, default=None, help='number of experiments')
parser.add_argument('--input_mode', type=str, default='whole',
                    choices=['whole', 'part'], help='input setting')
parser.add_argument('--input_size', nargs='+', default=16, type=int)
parser.add_argument('--overlap_size', type=int, default=16,
                    help='size of overlap')
parser.add_argument('--ignore_label', type=int, default=255,
                    help='ignore label')
args = parser.parse_args()

############# load dataset(indian_pines & pavia_univ...)######################

a=load()

All_data,labeled_data,rows_num,categories,r,c,flag=a.load_data(flag=args.flag)

print('Data has been loaded successfully!')

##################### normlization ######################

mi = 0
ma = 1
a = product(c, flag, 0, All_data)
All_data_norm = a.normlization(All_data[:, 1:-1], mi, ma)

print('Image normlization successfully!')

########################### Data preparation ##################################

if args.input_mode=='whole':

    X_data=All_data_norm.reshape(1,r,c,-1)

    args.print_freq=1

    args.input_size = (r, c)

elif args.input_mode=='part':


    image_size=(r, c)

    input_size=args.input_size

    LyEnd,LxEnd = np.subtract(image_size, input_size)

    Lx = np.linspace(0, LxEnd, np.ceil(LxEnd/np.float(input_size[1]-args.overlap_size))+1, endpoint=True).astype('int')
    Ly = np.linspace(0, LyEnd, np.ceil(LyEnd/np.float(input_size[0]-args.overlap_size))+1, endpoint=True).astype('int')

    image_3D=All_data_norm.reshape(r,c,-1)

    N=len(Ly)*len(Lx)

    X_data=np.zeros([N,input_size[0],input_size[1],image_3D.shape[-1]])#N,H,W,C

    i=0
    for j in range(len(Ly)):
        for k in range(len(Lx)):
            rStart,cStart = (Ly[j],Lx[k])
            rEnd,cEnd = (rStart+input_size[0],cStart+input_size[1])
            X_data[i] = image_3D[rStart:rEnd,cStart:cEnd,:]
            i+=1
else:
    raise NotImplementedError

print('{} image preparation Finished!, Data Number {}, '
      'Data size ({},{})'.format(args.flag,X_data.shape[0],X_data.shape[1],X_data.shape[2]))

X_data = torch.from_numpy(X_data.transpose(0, 3, 1, 2))#N,C,H,W

##################################### trn/val/tes ####################################

#Experimental memory
Experiment_result=np.zeros([categories+5,12])#OA,AA,kappa,trn_time,tes_time

# 实验次数
Experiment_num = args.exp_num

path = 'save'

if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(os.path.join(path, str(flag))):
    os.mkdir(os.path.join(path, str(flag)))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./save/' + str(flag) + '/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#kappa
kappa=0

y_map=All_data[:, -1].reshape(r,c)

# block_layer_acc_matrix=np.zeros([2,5])
#
# for jj in range(1,6):
#     args.layer_num = jj
#
#     print('block_num',args.block_num,'layer_num',args.layer_num)

best_OA = 0

for count in range(0, Experiment_num):

    a = product(c, flag, count, All_data)

    rows_num,trn_num,val_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num)

    #################################### trn_label #####################################

    y_trn_map=a.production_label(trn_num, y_map, split='Trn')

    if args.input_mode == 'whole':

        y_trn_data=y_trn_map.reshape(1,r,c)

    elif args.input_mode=='part':

        y_trn_data = np.zeros([N, input_size[0], input_size[1]], dtype=np.int32)  # N,H,W

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = Ly[j], Lx[k]
                rEnd, cEnd = rStart + input_size[0], cStart + input_size[1]
                y_trn_data[i] = y_trn_map[rStart:rEnd, cStart:cEnd]
                i+=1
    else:
        raise NotImplementedError

    y_trn_data-=1

    y_trn_data[y_trn_data<0]=args.ignore_label

    y_trn_data = torch.from_numpy(y_trn_data)

    print('Experiment {}，training dataset preparation Finished!'.format(count))

    #################################### val_label #####################################

    y_val_map = a.production_label(val_num, y_map, split='Val')


    if args.input_mode == 'whole':

        y_val_data = y_val_map.reshape(1, r, c)

    elif args.input_mode == 'part':

        y_val_data = np.zeros([N, input_size[0], input_size[1]])  # N,H,W

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = (Ly[j], Lx[k])
                rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                y_val_data[i,:,:] = y_val_map[rStart:rEnd, cStart:cEnd]
                i+=1
    else:
        raise NotImplementedError

    y_val_data -= 1

    y_val_data[y_val_data < 0] = args.ignore_label

    y_val_data = torch.from_numpy(y_val_data)

    print('Experiment {}，validation dataset preparation Finished!'.format(count))

    #################################### Searching #####################################

    torch.cuda.empty_cache()#GPU memory released

    trn_dataset=TensorDataset(X_data,y_trn_data)

    trn_loader=DataLoader(trn_dataset,batch_size=1,num_workers=0,
                          shuffle=True, drop_last=True, pin_memory=True)

    val_dataset = TensorDataset(X_data, y_val_data)

    val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False, pin_memory=True)

    # config lr & epoch

    srh_lr = 1e-2
    srh_epoch = 100

    net = hk3dseg_srh(X_data.shape[1], categories-1, args.block_num, args.layer_num, init_weights=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    logging.info("Srh param size = %fMB", count_parameters_in_MB(net))

    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=srh_lr, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(srh_epoch))

    loss_srh = []

    srh_time1 = time.time()

    a = operate()

    best_oa = 0

    for i in range(0, srh_epoch):
        loss_srh = a.train(args, i, srh_epoch, loss_srh, net, optimizer, scheduler, trn_loader, criterion, categories)
        if i % 10==0:
            val_OA = a.validation(args, net, val_loader, categories)

            logging.info('Experiment {}, srh_epoch {}, valid OA={}'.format(count, i, val_OA))

    srh_time2 = time.time()

    loss_srh = np.array(loss_srh)

    network_arch = np.array(net.network_structure())

    logging.info('Experiment {}, genotype'.format(count))
    logging.info(str(network_arch))
    logging.info('Experiment {}, searching finished!'.format(count))

    #################################### training #####################################

    trn_dataset = TensorDataset(X_data, y_trn_data)

    trn_loader = DataLoader(trn_dataset, batch_size=1, num_workers=0,
                            shuffle=True, drop_last=True, pin_memory=True)

    val_dataset = TensorDataset(X_data, y_val_data)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # config lr & epoch

    trn_lr = 1e-2
    trn_epoch = 300

    net = hk3dseg_evl(X_data.shape[1], categories - 1, args.block_num, args.layer_num, network_arch,
                      init_weights=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    logging.info("Trn param size = %fMB", count_parameters_in_MB(net))

    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=trn_lr, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(trn_epoch))

    loss_trn = []

    trn_time1 = time.time()

    a = operate()

    best_oa = 0

    for i in range(0, trn_epoch):
        loss_trn = a.train(args, i, trn_epoch, loss_trn, net, optimizer, scheduler, trn_loader, criterion, categories)
        if i % 10 == 0:
            val_OA = a.validation(args, net, val_loader, categories)

            logging.info('Experiment {}, trn_epoch {}, valid OA={}'.format(count, i, val_OA))

    trn_time2 = time.time()

    loss_trn = np.array(loss_trn)

    logging.info('Experiment {}, training finished!'.format(count))

    ####################################### inference ####################################

    a = product(c, flag, count, All_data)

    y_tes_map = a.production_label(tes_num, y_map, split='Tes')

    y_tes_data = y_tes_map.reshape(r, c)

    y_tes_data -= 1

    y_tes_data[y_tes_data < 0] = 255

    print('Experiment {}，Testing dataset preparation Finished!'.format(count))

    tes_time1 = time.time()

    if args.input_mode == 'whole':

        net.eval()
        with torch.no_grad():
            pred = net(Variable(X_data.float()).cuda(non_blocking=True))
            pred = pred.cpu().numpy()
            y_tes_pred = np.argmax(pred, 1).squeeze(0)

    elif args.input_mode == 'part':

        img=torch.from_numpy(image_3D).permute(2,0,1) #C,H,W
        y_tes_pred = np.zeros([r, c])
        net.eval()

        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = (Ly[j], Lx[k])
                rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                img_part = img[:,rStart:rEnd,cStart:cEnd].unsqueeze(0)
                with torch.no_grad():
                    pred = net(Variable(img_part.float()).cuda(non_blocking=True))
                pred = pred.cpu().numpy()
                pred = np.argmax(pred,1).squeeze(0)

                if j == 0 and k == 0:
                    y_tes_pred[rStart:rEnd, cStart:cEnd] = pred
                elif j == 0 and k > 0:
                    y_tes_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = pred[:,
                                                                                       int(args.overlap_size / 2):]
                elif j > 0 and k == 0:
                    y_tes_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = pred[
                                                                                       int(args.overlap_size / 2):,
                                                                                       :]
                else:
                    y_tes_pred[rStart + int(args.overlap_size / 2):rEnd,
                    cStart + int(args.overlap_size / 2):cEnd] = pred[int(args.overlap_size / 2):,
                                                                int(args.overlap_size / 2):]
    else:
        raise NotImplementedError

    tes_time2 = time.time()

    logging.info('########### Experiment {}，Model Testing Period Finished! ############'.format(count))

    ####################################### assess ###########################################

    y_tes_data_1d = y_tes_data.reshape(r*c)
    y_tes_pred_1d = y_tes_pred.reshape(r*c)

    y_tes_gt=y_tes_data_1d[tes_num]
    y_tes=y_tes_pred_1d[tes_num]

    logging.info('==================Test set=====================')
    logging.info('Experiment {}，Testing set OA={}'.format(count,np.mean(y_tes_gt==y_tes)))
    logging.info('Experiment {}，Testing set Kappa={}'.format(count,cohen_kappa_score(y_tes_gt,y_tes)))

    OA = np.mean(y_tes_gt == y_tes)
    Kappa = cohen_kappa_score(y_tes_gt, y_tes)


    best_arch = 0

    if OA > best_OA:
        best_OA = OA
        best_arch = network_arch
        torch.save(net.state_dict(), './save/{}/hk3dseg_{}.pth'.format(flag, flag))
        np.save('./save/{}/trn_num.npy'.format(flag), trn_num)
        np.save('./save/{}/pre_num.npy'.format(flag), pre_num)
        np.save('./save/{}/best_arch.npy'.format(flag), best_arch)

    ## Detailed information (every class accuracy)

    num_tes=np.zeros([categories-1])
    num_tes_pred=np.zeros([categories-1])
    for k in y_tes_gt:
        num_tes[int(k)]+=1# class index start from 0
    for j in range(y_tes_gt.shape[0]):
        if y_tes_gt[j]==y_tes[j]:
            num_tes_pred[int(y_tes_gt[j])]+=1

    Acc=num_tes_pred/num_tes*100

    Experiment_result[0,count]=OA * 100 #OA
    Experiment_result[1,count]=np.mean(Acc) #AA
    Experiment_result[2,count]= Kappa * 100 #Kappa
    Experiment_result[3, count] = srh_time2 - srh_time1
    Experiment_result[4, count] = trn_time2 - trn_time1
    Experiment_result[4, count] = tes_time2 - tes_time1
    Experiment_result[6:,count]=Acc

    logging.info('Experiment {}, genotype:'.format(count))
    logging.info(str(network_arch))
    logging.info('Experiment {}, Testing set OA={:.4f}'.format(count, OA * 100))
    logging.info('Experiment {}, Testing set Kappa={:.4f}'.format(count, Kappa * 100))

    for i in range(categories - 1):
        print('Class_{}: accuracy {:.4f}.'.format(i + 1, Acc[i]))

    print('########### Experiment {}，Model assessment Finished！ ###########'.format(count))

    scio.savemat('./save/' + str(flag) + '/hk3dseg_result_' + str(flag) + '.mat', {'data': Experiment_result})

scio.savemat('./save/'+str(flag)+'/hk3dseg_loss_srh_'+str(flag)+'.mat',{'hk3dseg_loss_srh':loss_srh})
scio.savemat('./save/'+str(flag)+'/hk3dseg_loss_trn_'+str(flag)+'.mat',{'hk3dseg_loss_trn':loss_trn})

# block_layer_acc_matrix[0, jj - 1] = np.mean(Experiment_result[[0],:-2],axis=1)
# block_layer_acc_matrix[1, jj - 1] = np.std(Experiment_result[[0],:-2],axis=1)
# scio.savemat('./save/'+str(flag)+'/hk3dseg_blk_layer_acc_'+str(flag)+'.mat',{'data':block_layer_acc_matrix})

########## mean value & standard deviation #############

Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

logging.info('BEST genotype:')
logging.info(str(best_arch))

scio.savemat('./save/' + str(flag) + '/hk3dseg_result_' + str(flag) + '.mat', {'data': Experiment_result})

################################ Classification map ######################################

trn_num = np.load('./save/{}/trn_num.npy'.format(flag))
pre_num = np.load('./save/{}/pre_num.npy'.format(flag))
best_arch = np.load('./save/{}/best_arch.npy'.format(flag))

net = hk3dseg_evl(X_data.shape[1], categories - 1, args.block_num, args.layer_num, best_arch, init_weights=True)

net.load_state_dict(torch.load('./save/{}/hk3dseg_{}.pth'.format(flag, flag), map_location='cpu'))

net = net.cuda()

y_trn = All_data[trn_num, -1]

pre_time1 = time.time()

if args.input_mode == 'whole':

    net.eval()
    with torch.no_grad():
        pred = net(Variable(X_data.float()).cuda(non_blocking=True))
        pred = pred.cpu().numpy()
        y_pred = np.argmax(pred, 1).squeeze(0)

elif args.input_mode == 'part':

    img = torch.from_numpy(image_3D).permute(2, 0, 1)  # C,H,W
    y_pred = np.zeros([r, c])
    net.eval()

    for j in range(len(Ly)):
        for k in range(len(Lx)):
            rStart, cStart = (Ly[j], Lx[k])
            rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
            img_part = img[:, rStart:rEnd, cStart:cEnd].unsqueeze(0)
            with torch.no_grad():
                pred = net(Variable(img_part.float()).cuda(non_blocking=True))
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, 1).squeeze(0)

            if j == 0 and k == 0:
                y_pred[rStart:rEnd, cStart:cEnd] = pred
            elif j == 0 and k > 0:
                y_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = pred[:,
                                                                                    int(args.overlap_size / 2):]
            elif j > 0 and k == 0:
                y_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = pred[
                                                                                    int(args.overlap_size / 2):,
                                                                                    :]
            else:
                y_pred[rStart + int(args.overlap_size / 2):rEnd,
                cStart + int(args.overlap_size / 2):cEnd] = pred[int(args.overlap_size / 2):,
                                                            int(args.overlap_size / 2):]
else:
    raise NotImplementedError

pre_time2 = time.time()

print('prediction finished！cost {} secs'.format(pre_time2-pre_time1))

#############  展示 ##############

y_disp = np.zeros([All_data.shape[0]])
y_pred = y_pred.reshape(-1)

y_disp_all = y_disp.copy()
y_disp_all[trn_num] = y_trn
y_disp_all[pre_num] = y_pred[pre_num] + 1

#####################  保存预测全图 #####################

#np.save('./save/' + str(flag) + '/hd3dseg_all_' + str(flag) + '.npy', y_disp_all.reshape(r, c) - 1)


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
pred.save('./save/'+str(flag)+'/hk3dseg_all_'+str(flag)+'.png')

print('Classification map saving finished')
