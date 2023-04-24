import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from func import intersectionAndUnionGPU

class hk3dlayer_srh(nn.Module):
    def __init__(self, in_channel):
        super(hk3dlayer_srh, self).__init__()

        self.W1 = nn.Parameter(torch.randn(1, 1, 9, 1, 1))
        self.W2 = nn.Parameter(torch.randn(in_channel // 4, 1, 9, 9))
        self.relu = nn.ReLU(inplace=True)
        self.gns = nn.GroupNorm(16,in_channel // 4)

        self.conv1 = nn.Conv2d(in_channel, in_channel//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel//4, in_channel, kernel_size=1)
        self.gn1 = nn.GroupNorm(16,in_channel//4)
        self.gn2 = nn.GroupNorm(16,in_channel)

    def forward(self,x):

        res = x

        x = self.gn1(self.conv1(x)) #b, c, h, w
        x_sa = x
        b, c, h, w = x.shape
        x = x.reshape(b, 1, c, h, w)

        ######### spectral

        ## circle

        self.mask_11 = torch.zeros(self.W1.shape).cuda()
        self.mask_11[:, :, 3:-3] = 1

        self.mask_12 = torch.zeros(self.W1.shape).cuda()
        self.mask_12[:, :, 2:-2] = 1

        self.mask_13 = torch.zeros(self.W1.shape).cuda()
        self.mask_13[:, :, 1:-1] = 1

        self.mask_14 = torch.ones(self.W1.shape).cuda()

        W1_circle1 = torch.mul(self.W1, self.mask_12 - self.mask_11)
        W1_circle2 = torch.mul(self.W1, self.mask_13 - self.mask_12)
        W1_circle3 = torch.mul(self.W1, self.mask_14 - self.mask_13)

        ## structure

        self.a1 = torch.zeros(4).cuda()

        self.a1[0] = torch.sum(torch.mul(self.W1, self.mask_11), (2,3,4,1,0)) / (torch.sum(self.mask_11,(2,3,4,1,0)) + 1e-10)
        self.a1[1] = torch.sum(W1_circle1, (2,3,4,1,0)) / (torch.sum(self.mask_12 - self.mask_11,(2,3,4,1,0)) + 1e-10)
        self.a1[2] = torch.sum(W1_circle2, (2,3,4,1,0)) / (torch.sum(self.mask_13 - self.mask_12, (2,3,4,1,0)) + 1e-10)
        self.a1[3] = torch.sum(W1_circle3, (2,3,4,1,0)) / (torch.sum(self.mask_14 - self.mask_13, (2,3,4,1,0)) + 1e-10)

        self.a1 = F.softmax(self.a1,dim=0)

        #a = a.detach()

        O1=[F.conv3d(x, torch.mul(self.W1, self.mask_11), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_12), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_13), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_14), stride=1, padding=(4,0,0))]

        x = self.a1[0] * O1[0] + self.a1[1] * O1[1] + \
            self.a1[2] * O1[2] + self.a1[3] * O1[3]

        x = x.squeeze(1)

        ######### spatial

        ## circle

        self.mask_21 = torch.zeros(self.W2.shape).cuda()
        self.mask_21[:, :, 3:-3, 3:-3] = 1

        self.mask_22 = torch.zeros(self.W2.shape).cuda()
        self.mask_22[:, :, 2:-2, 2:-2] = 1

        self.mask_23 = torch.zeros(self.W2.shape).cuda()
        self.mask_23[:, :, 1:-1, 1:-1] = 1

        self.mask_24 = torch.ones(self.W2.shape).cuda()

        W2_circle1 = torch.mul(self.W2, self.mask_22 - self.mask_21)
        W2_circle2 = torch.mul(self.W2, self.mask_23 - self.mask_22)
        W2_circle3 = torch.mul(self.W2, self.mask_24 - self.mask_23)

        ## structure

        self.a2 = torch.zeros(4).cuda()

        self.a2[0] = torch.sum(torch.mul(self.W2, self.mask_21), (3, 2, 1, 0)) / (
                    torch.sum(self.mask_21, (3, 2, 1, 0)) + 1e-10)
        self.a2[1] = torch.sum(W2_circle1, (3, 2, 1, 0)) / (
                    torch.sum(self.mask_22 - self.mask_21, (3, 2, 1, 0)) + 1e-10)
        self.a2[2] = torch.sum(W2_circle2, (3, 2, 1, 0)) / (
                    torch.sum(self.mask_23 - self.mask_22, (3, 2, 1, 0)) + 1e-10)
        self.a2[3] = torch.sum(W2_circle3, (3, 2, 1, 0)) / (
                    torch.sum(self.mask_24 - self.mask_23, (3, 2, 1, 0)) + 1e-10)

        self.a2 = F.softmax(self.a2, dim=0)

        b,cc,h,w = x_sa.shape

        O2 = [F.conv2d(x_sa, torch.mul(self.W2, self.mask_21), stride=1, padding=(4, 4),groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_22), stride=1, padding=(4, 4),groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_23), stride=1, padding=(4, 4),groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_24), stride=1, padding=(4, 4),groups=cc)]

        x_sa = self.a2[0] * O2[0] + self.a2[1] * O2[1] + \
            self.a2[2] * O2[2] + self.a2[3] * O2[3]

        x = self.relu(self.gns(x + x_sa))

        x = self.relu(self.gn2(self.conv2(x)))

        x = x + res

        return x

    def layer_structure(self):
        idx1 = torch.argmax(self.a1)
        idx2 = torch.argmax(self.a2)
        idx = 10*idx1 + idx2
        return idx.cpu().numpy()

class hk3dblock_srh(nn.Module):

    def __init__(self, in_channel, out_channel, layer_num):
        super(hk3dblock_srh, self).__init__()

        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(hk3dlayer_srh(out_channel))

    def forward(self, x):

        x = self.conv(x)

        for i in range(self.layer_num):
            x = self.layers[i](x)

        return x
    def block_structure(self):
        block_arch = []
        for i in range(self.layer_num):
            block_arch.append(self.layers[i].layer_structure())
        return block_arch


class hk3dseg_srh(nn.Module):
    def __init__(self, spec_band, num_classes, block_num, layer_num,init_weights=True):
        super(hk3dseg_srh, self).__init__()

        depth = 64
        self.conv0 = nn.Conv2d(spec_band, depth, kernel_size=1, stride=1, padding=0, bias=False)

        in_channel = depth
        out_channel = 32

        self.block_num = block_num
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            out_channel = out_channel * 2
            print(in_channel, out_channel)
            self.blocks.append(hk3dblock_srh(in_channel, out_channel, layer_num))
            in_channel = out_channel

        self.conv1 = nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        _, _, h, w = x.shape

        x = self.conv0(x)

        for i in range(self.block_num):
            x = self.blocks[i](x)
            if i==0:
                x = self.pool(x)

        x = self.conv1(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

    def network_structure(self):
        network_arch = []
        for i in range(self.block_num):
            network_arch.append(self.blocks[i].block_structure())
        return network_arch

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data, gain=1)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

#################################### evaluation ###########################################

class hk3dlayer_evl(nn.Module):
    def __init__(self, in_channel, layer_arch):
        super(hk3dlayer_evl, self).__init__()

        self.W1 = nn.Parameter(torch.randn(1, 1, 9, 1, 1))

        self.mask_11 = torch.zeros(self.W1.shape).cuda()
        self.mask_11[:, :, 3:-3] = 1

        self.mask_12 = torch.zeros(self.W1.shape).cuda()
        self.mask_12[:, :, 2:-2] = 1

        self.mask_13 = torch.zeros(self.W1.shape).cuda()
        self.mask_13[:, :, 1:-1] = 1

        self.mask_14 = torch.ones(self.W1.shape).cuda()

        self.W2 = nn.Parameter(torch.randn(in_channel // 4, 1, 9, 9))

        self.mask_21 = torch.zeros(self.W2.shape).cuda()
        self.mask_21[:, :, 3:-3, 3:-3] = 1

        self.mask_22 = torch.zeros(self.W2.shape).cuda()
        self.mask_22[:, :, 2:-2, 2:-2] = 1

        self.mask_23 = torch.zeros(self.W2.shape).cuda()
        self.mask_23[:, :, 1:-1, 1:-1] = 1

        self.mask_24 = torch.ones(self.W2.shape).cuda()

        self.relu = nn.ReLU(inplace=True)

        self.gnm = nn.GroupNorm(16, in_channel // 4)

        self.conv1 = nn.Conv2d(in_channel, in_channel//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel//4, in_channel, kernel_size=1)

        self.gn1 = nn.GroupNorm(16, in_channel//4)
        self.gn2 = nn.GroupNorm(16, in_channel)

        self.idx1 = int(layer_arch // 10)
        self.idx2 = int(layer_arch % 10)

    def forward(self,x):

        res = x

        x = self.gn1(self.conv1(x)) #b, c, h, w

        x_sa = x

        b, c, h, w = x.shape
        x = x.reshape(b, 1, c, h, w)

        ######### spectral

        O1=[F.conv3d(x, torch.mul(self.W1, self.mask_11), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_12), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_13), stride=1, padding=(4,0,0)),
           F.conv3d(x, torch.mul(self.W1, self.mask_14), stride=1, padding=(4,0,0))]

        x = O1[self.idx1]

        x = x.squeeze(1)

        ######### spatial

        b, cc, h, w = x_sa.shape

        O2 = [F.conv2d(x_sa, torch.mul(self.W2, self.mask_21), stride=1, padding=(4, 4), groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_22), stride=1, padding=(4, 4), groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_23), stride=1, padding=(4, 4), groups=cc),
             F.conv2d(x_sa, torch.mul(self.W2, self.mask_24), stride=1, padding=(4, 4), groups=cc)]

        x_sa = O2[self.idx2]

        x = self.relu(self.gnm(x+x_sa))

        x = self.relu(self.gn2(self.conv2(x)))

        x = x + res

        return x

class hk3dblock_evl(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, block_arch):
        super(hk3dblock_evl, self).__init__()

        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(hk3dlayer_evl(out_channel, block_arch[i]))

    def forward(self, x):

        x = self.conv(x)

        for i in range(self.layer_num):
            x = self.layers[i](x)

        return x

class hk3dseg_evl(nn.Module):
    def __init__(self, spec_band, num_classes, block_num, layer_num, arch, init_weights=True):
        super(hk3dseg_evl, self).__init__()

        depth = 64
        self.conv0 = nn.Conv2d(spec_band, depth, kernel_size=1, stride=1, padding=0, bias=False)

        in_channel = depth
        out_channel = 32

        self.block_num = block_num
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            out_channel = out_channel * 2
            print(in_channel, out_channel)
            self.blocks.append(hk3dblock_evl(in_channel,out_channel,layer_num, arch[i]))
            in_channel = out_channel

        self.conv1 = nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        _ ,_ ,h, w = x.shape

        x = self.conv0(x)

        for i in range(self.block_num):
            x = self.blocks[i](x)
            if i==0:
                x = self.pool(x)

        x = self.conv1(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

        # fork from https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data, gain=1)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class operate():

    def train(self, args, epoch, epochs, loss_trn, net, optimizer, scheduler, trn_loader, criterion, categories):
        net.train()  # train mode
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        for idx, (X_data, y_target) in enumerate(trn_loader):

            X_data=Variable(X_data.float()).cuda(non_blocking=True)
            y_target = Variable(y_target.float().long()).cuda(non_blocking=True)

            y_pred = net.forward(X_data)

            loss = criterion(y_pred, y_target)

            _, predicted = torch.max(y_pred, 1)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            n = X_data.size(0)  # batch size
            loss_meter.update(loss.item(), n)
            intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories-1, args.ignore_label)
            intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), target_meter.update(target)

        scheduler.step()

        loss_trn.append(loss_meter.avg)

        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        print('Training epoch [{}/{}]: Loss {:.4f} AA/OA {:.4f}/{:.4f}.'.format(epoch + 1,
                                                                    epochs,loss_meter.avg,
                                                                    mAcc, allAcc))
        return loss_trn

    def validation(self, args, net, val_loader, categories):
        print('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
        net.eval()  # evaluation mode
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        for idx, (X_data, y_target) in enumerate(val_loader):
            with torch.no_grad():
                X_data = Variable(X_data.float()).cuda(non_blocking=True)
                y_target = Variable(y_target.float().long()).cuda(non_blocking=True)
                y_pred = net.forward(X_data)

            _, predicted = torch.max(y_pred, 1)

            # compute acc
            intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories - 1, args.ignore_label)
            intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), target_meter.update(target)

        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        print('Validation AA/OA {:.4f}/{:.4f}.'.format(mAcc, allAcc))

        for i in range(categories-1):
            print('Class_{}: accuracy {:.4f}.'.format(i+1, accuracy_class[i]))

        print('>>>>>>>>>>>>>>>> End Evaluation <<<<<<<<<<<<<<<<<<')

        return allAcc