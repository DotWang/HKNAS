import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

## search

class hklayer_srh(nn.Module):
    def __init__(self, in_channel):
        super(hklayer_srh, self).__init__()

        self.W = nn.Parameter(torch.randn(in_channel//4,in_channel//4,9))
        self.relu = nn.ReLU(inplace=True)

        self.bns = nn.BatchNorm1d(in_channel//4)

        self.conv1 = nn.Conv2d(in_channel, in_channel//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel//4, in_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channel//4)
        self.bn2 = nn.BatchNorm1d(in_channel)

    def forward(self,x):

        res = x

        # 1*1 conv
        b, c, h = x.shape
        x = x.reshape(b, c, h, 1)
        x = self.bn1(self.conv1(x).reshape(b, -1, h))

        ## circle

        self.mask_1 = torch.zeros(self.W.shape).cuda()
        self.mask_1[:, :, 3:-3] = 1

        self.mask_2 = torch.zeros(self.W.shape).cuda()
        self.mask_2[:, :, 2:-2] = 1

        self.mask_3 = torch.zeros(self.W.shape).cuda()
        self.mask_3[:, :, 1:-1] = 1

        self.mask_4 = torch.ones(self.W.shape).cuda()

        W_circle1 = torch.mul(self.W, self.mask_2 - self.mask_1)
        W_circle2 = torch.mul(self.W, self.mask_3 - self.mask_2)
        W_circle3 = torch.mul(self.W, self.mask_4 - self.mask_3)

        ## structure

        self.a = torch.zeros(4).cuda()

        self.a[0] = torch.sum(torch.mul(self.W, self.mask_1), (2,1,0)) / (torch.sum(self.mask_1,(2,1,0)) + 1e-10)
        self.a[1] = torch.sum(W_circle1, (2, 1, 0)) / (torch.sum(self.mask_2 - self.mask_1,(2, 1,0)) + 1e-10)
        self.a[2] = torch.sum(W_circle2, (2, 1, 0)) / (torch.sum(self.mask_3 - self.mask_2, (2, 1,0)) + 1e-10)
        self.a[3] = torch.sum(W_circle3, (2, 1, 0)) / (torch.sum(self.mask_4 - self.mask_3, (2, 1, 0)) + 1e-10)

        self.a = F.softmax(self.a,dim=0)

        #a = a.detach()

        O=[F.conv1d(x, torch.mul(self.W, self.mask_1), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_2), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_3), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_4), stride=1, padding=4)]

        #if self.training:

        x = self.a[0] * O[0] + self.a[1] * O[1] + \
            self.a[2] * O[2] + self.a[3] * O[3]

        x = self.relu(self.bns(x))

        # else:
        #     idx = torch.argmax(a)
        #     x = self.bns[idx](O[idx])
            # print('select conv {}*{}'.format(int(2*idx+3),int(2*idx+3)))

        # 1*1 conv
        b, c, h = x.shape
        x = x.reshape(b, c, h, 1)
        x = self.relu(self.bn2(self.conv2(x).reshape(b, -1, h)))

        x = x + res

        return x

    def layer_structure(self):
        idx = torch.argmax(self.a)
        return idx.cpu().numpy()

class hkblock_srh(nn.Module):

    def __init__(self, in_channel, out_channel, layer_num):
        super(hkblock_srh, self).__init__()

        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(hklayer_srh(out_channel))

    def forward(self, x):
        b,c,h=x.shape
        x = x.reshape(b,c,h,1)
        x = self.conv(x).reshape(b,-1,h)

        for i in range(self.layer_num):
            x = self.layers[i](x)

        return x
    def block_structure(self):
        block_arch = []
        for i in range(self.layer_num):
            block_arch.append(self.layers[i].layer_structure())
        return block_arch


class hk1dcls_srh(nn.Module):
    def __init__(self, spec_band, num_classes, block_num, layer_num, init_weights=True):
        super(hk1dcls_srh, self).__init__()

        self.fc0 = nn.Linear(spec_band,96)

        in_channel = 1
        out_channel = 64

        self.block_num = block_num
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            #out_channel = out_channel*2
            if i in [self.block_num//4, 2*self.block_num//4, 3*self.block_num//4]:
                out_channel = out_channel*2
                print(in_channel, out_channel)
            self.blocks.append(hkblock_srh(in_channel,out_channel,layer_num))
            in_channel = out_channel

        self.pool = nn.AvgPool1d(2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(out_channel, num_classes).float()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.fc0(x)

        x = x.reshape(-1,1,x.shape[-1])

        feas = []

        for i in range(self.block_num):
            x = self.blocks[i](x)
            # feas.append(out)
            # x = sum(feas)
            if i in [self.block_num//4, 2*self.block_num//4, 3*self.block_num//4]:
                x = self.pool(x)

        x = self.avgpool(x).view(x.size(0), -1)

        score = self.fc1(x)

        return score

    def network_structure(self):
        network_arch = []
        for i in range(self.block_num):
            network_arch.append(self.blocks[i].block_structure())
        return network_arch

    # fork from https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data, gain=1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

#################################### evaluation ###########################################

class hklayer_evl(nn.Module):
    def __init__(self, in_channel, layer_arch):
        super(hklayer_evl, self).__init__()

        self.W = nn.Parameter(torch.randn(in_channel//4,in_channel//4,9))

        self.mask_1 = torch.zeros(self.W.shape).cuda()
        self.mask_1[:, :, 3:-3] = 1

        self.mask_2 = torch.zeros(self.W.shape).cuda()
        self.mask_2[:, :, 2:-2] = 1

        self.mask_3 = torch.zeros(self.W.shape).cuda()
        self.mask_3[:, :, 1:-1] = 1

        self.mask_4 = torch.ones(self.W.shape).cuda()

        self.relu = nn.ReLU(inplace=True)
        self.bns = nn.ModuleList()

        self.bn = nn.BatchNorm1d(in_channel//4)

        self.conv1 = nn.Conv2d(in_channel, in_channel//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel//4, in_channel, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(in_channel // 4)
        self.bn2 = nn.BatchNorm1d(in_channel)

        self.idx = layer_arch

    def forward(self,x):

        res = x

        # 1*1 conv
        b, c, h = x.shape
        x = x.reshape(b, c, h, 1)
        x = self.bn1(self.conv1(x).reshape(b, -1, h))

        O=[F.conv1d(x, torch.mul(self.W, self.mask_1), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_2), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_3), stride=1, padding=4),
           F.conv1d(x, torch.mul(self.W, self.mask_4), stride=1, padding=4)]

        x = self.relu(self.bn(O[self.idx]))

        # 1*1 conv
        b, c, h = x.shape
        x = x.reshape(b, c, h, 1)
        x = self.relu(self.bn2(self.conv2(x).reshape(b, -1, h)))

        x = x + res

        return x

class hkblock_evl(nn.Module):

    def __init__(self, in_channel, out_channel, layer_num, block_arch):
        super(hkblock_evl, self).__init__()

        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(hklayer_evl(out_channel, block_arch[i]))


    def forward(self, x):
        b,c,h=x.shape
        x = x.reshape(b,c,h,1)
        x = self.conv(x).reshape(b,-1,h)

        for i in range(self.layer_num):
            x = self.layers[i](x)

        return x


class hk1dcls_evl(nn.Module):
    def __init__(self, spec_band, num_classes, block_num, layer_num, arch, init_weights=True):
        super(hk1dcls_evl, self).__init__()

        self.fc0 = nn.Linear(spec_band, 96)

        in_channel = 1
        out_channel = 64

        self.block_num = block_num
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            #out_channel = out_channel*2
            if i in [self.block_num//4, 2*self.block_num//4, 3*self.block_num//4]:
                out_channel = out_channel*2
                print(in_channel, out_channel)
            self.blocks.append(hkblock_evl(in_channel,out_channel,layer_num, arch[i]))
            in_channel = out_channel

        self.pool = nn.AvgPool1d(2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(out_channel, num_classes).float()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.fc0(x)

        x = x.reshape(-1,1,x.shape[-1])

        feas = []

        for i in range(self.block_num):
            x = self.blocks[i](x)
            # feas.append(out)
            # x = sum(feas)
            if i in [self.block_num//4, 2*self.block_num//4, 3*self.block_num//4]:
                x = self.pool(x)

        x = self.avgpool(x).view(x.size(0), -1)

        score = self.fc1(x)

        return score

    # fork from https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data, gain=1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class operate():
    def train(self, epoch, loss_trn, net, optimizer, scheduler, trn_loader, criterion):
        net.train()  # train mode
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spat, y_target) in enumerate(trn_loader):
            X_spat = Variable(X_spat.float()).cuda()
            ######GPU
            y_target = Variable(y_target.float().long()).cuda()
            y_pred = net.forward(X_spat)
            loss = criterion(y_pred, y_target)

            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            # print(torch.unique(predicted))
            # print(torch.unique(y_target))
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if idx % 20==0:

            del X_spat, y_target
            del y_pred
            # del loss
        scheduler.step()
        loss_trn.append(epochavg_loss / (idx + 1))
        print('train epoch:{},train loss:{},correct/total:{:.4f}%'.format(epoch,
               epochavg_loss / (idx + 1),100 * correct.item() / total))
        #loss_trn.append(epochavg_loss / (idx + 1))
        return loss_trn

    def inference(self,net, data_loader, criterion, MODE='VAL'):
        net.eval()  # evaluation mode
        inf_loss = 0
        num = 1
        correct = 0
        total = 0
        for idx, (X_spat, y_target) in enumerate(data_loader):
            with torch.no_grad():
                X_spat = Variable(X_spat.float()).cuda()#GPU
                y_target = Variable(y_target.float().long()).cuda()
                y_score = net.forward(X_spat)
            loss = criterion(y_score, y_target)
            inf_loss += loss.float()  # save memory

            _, predicted = torch.max(y_score.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            y_pred_inf = np.argmax(y_score.detach().cpu().numpy(), axis=1) + 1
            if num == 1:
                inf_result = y_pred_inf
            else:
                inf_result = np.hstack((inf_result, y_pred_inf))
            if idx % 20 == 0 and idx > 0 and MODE!='PRED':
                print('test loss:{},{}/{}({:.2f}%),correct/total:{:.4f}%'.format(
                    loss.item(), idx * X_spat.shape[0],len(data_loader.dataset),100 * idx * X_spat.shape[0] / len(
                    data_loader.dataset),100 * correct.item() / total))
            num += 1
            del X_spat, y_target
            del loss
            del y_score
            del y_pred_inf
        avg_inf_loss = inf_loss / len(data_loader.dataset)
        if MODE == 'VAL':
            print('Over all validation loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        elif MODE == 'TEST':
            print('Over all testing loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        elif MODE == 'PRED':
            pass
        else:
            raise NotImplementedError

        return inf_result