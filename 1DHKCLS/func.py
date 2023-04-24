import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class load():
    # load dataset(indian_pines & pavia_univ.)
    def load_data(self, flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('/project/DW/Dataset/Indian_pines.mat')
            Ind_pines_gt_dict = scio.loadmat('/project/DW/Dataset/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            # remove the water absorption bands

            no_absorption = list(set(np.arange(0, 103)) | set(np.arange(108, 149)) | set(np.arange(163, 219)))

            original = Ind_pines_dict['indian_pines'][:, :, no_absorption].reshape(145 * 145, 200)

            print(original.shape)
            print('Remove wate absorption bands successfully!')

            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines'].shape[0]
            c = Ind_pines_dict['indian_pines'].shape[1]
            categories = 17
        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('/project/DW/Dataset/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('/project/DW/Dataset/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10
        if flag == 'ksc':
            ksc_dict = scio.loadmat('/project/DW/Dataset/KSC.mat')
            ksc_gt_dict = scio.loadmat('/project/DW/Dataset/KSC_gt.mat')

            print(ksc_dict['KSC'].shape)
            print(ksc_gt_dict['KSC_gt'].shape)

            original = ksc_dict['KSC'].reshape(512 * 614, 176)
            original[original > 400] = 0
            gt = ksc_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = ksc_dict['KSC'].shape[0]
            c = ksc_dict['KSC'].shape[1]
            categories = 14
        if flag == 'salina':
            salinas_dict = scio.loadmat('/project/DW/Dataset/Salinas.mat')
            salinas_gt_dict = scio.loadmat('/project/DW/Dataset/Salinas_gt.mat')

            print(salinas_dict['salinas'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas'].reshape(512 * 217, 224)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas'].shape[0]
            c = salinas_dict['salinas'].shape[1]
            categories = 17
        if flag == 'houston':
            houst_dict = scio.loadmat('/project/DW/Dataset/Houston.mat')
            houst_gt_dict = scio.loadmat('/project/DW/Dataset/Houston_GT.mat')

            print(houst_dict['Houston'].shape)
            print(houst_gt_dict['Houston_GT'].shape)

            original = houst_dict['Houston'].reshape(349 * 1905, 144)
            gt = houst_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = houst_dict['Houston'].shape[0]
            c = houst_dict['Houston'].shape[1]
            categories = 16

        if flag == 'longkou':
            longkou_dict = scio.loadmat(
                '/project/DW/Dataset/WHU_Hi_LongKou.mat')
            longkou_gt_dict = scio.loadmat(
                '/project/DW/Dataset/WHU_Hi_LongKou_gt.mat')

            print(longkou_dict['WHU_Hi_LongKou'].shape)
            print(longkou_gt_dict['WHU_Hi_LongKou_gt'].shape)

            original = longkou_dict['WHU_Hi_LongKou'].reshape(550 * 400, 270)
            gt = longkou_gt_dict['WHU_Hi_LongKou_gt'].reshape(550 * 400, 1)

            r = longkou_dict['WHU_Hi_LongKou'].shape[0]
            c = longkou_dict['WHU_Hi_LongKou'].shape[1]
            categories = 10

        if flag == 'hanchuan':
            hanchuan_dict = scio.loadmat(
                '../../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat')
            hanchuan_gt_dict = scio.loadmat(
                '../../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat')

            print(hanchuan_dict['WHU_Hi_HanChuan'].shape)
            print(hanchuan_gt_dict['WHU_Hi_HanChuan_gt'].shape)

            original = hanchuan_dict['WHU_Hi_HanChuan'].reshape(1217 * 303, 274)
            gt = hanchuan_gt_dict['WHU_Hi_HanChuan_gt'].reshape(1217 * 303, 1)

            r = hanchuan_dict['WHU_Hi_HanChuan'].shape[0]
            c = hanchuan_dict['WHU_Hi_HanChuan'].shape[1]
            categories = 17

        if flag == 'honghu':
            honghu_dict = scio.loadmat(
                '../../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu.mat')
            honghu_gt_dict = scio.loadmat(
                '../../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat')

            print(honghu_dict['WHU_Hi_HongHu'].shape)
            print(honghu_gt_dict['WHU_Hi_HongHu_gt'].shape)

            original = honghu_dict['WHU_Hi_HongHu'].reshape(940 * 475, 270)
            gt = honghu_gt_dict['WHU_Hi_HongHu_gt'].reshape(940 * 475, 1)

            r = honghu_dict['WHU_Hi_HongHu'].shape[0]
            c = honghu_dict['WHU_Hi_HongHu'].shape[1]
            categories = 23

        rows = np.arange(gt.shape[0])  # 从0开始
        # 行号(ID)，特征数据，类别号
        All_data = np.c_[rows, original, gt]

        # 剔除非0类别，获取所有labeled数据
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # 所有labeled数据的ID

        return All_data, labeled_data, rows_num, categories, r, c, flag

class product():
    def __init__(self, flag, seed):
        self.flag = flag
        self.seed = seed
    # product the training and testing pixel ID
    def generation_num(self,labeled_data, rows_num, All_data):

        train_num = []
        valid_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]  # 某类别的所有ID
            #print(i, temp_num.shape[0])
            np.random.seed(self.seed)
            np.random.shuffle(temp_num)  # 打乱顺序
            if self.flag == 'indian':
                if i == 7:
                    train_num.append(temp_num[0:7])
                    valid_num.append(temp_num[7:14])
                elif i == 9:
                    train_num.append(temp_num[0:5])
                    valid_num.append(temp_num[5:10])
                else:
                    train_num.append(temp_num[0:10])
                    valid_num.append(temp_num[10:20])
            else:
                train_num.append(temp_num[0:10])
                valid_num.append(temp_num[10:20])

        trn_num = [x for j in train_num for x in j]  # 合并list中各元素
        val_num = [x for j in valid_num for x in j]
        tes_num = list(set(rows_num) - set(trn_num) - set(val_num))
        pre_num = list(set(range(0, All_data.shape[0])) - set(trn_num))
        print('number of training sample', len(trn_num))
        print('number of validation sample', len(val_num))
        return list(map(int, rows_num)), list(map(int, trn_num)), list(map(int, val_num)), list(map(int, tes_num)), list(map(int, pre_num))

    def normlization(self, data_spat, mi, ma):

        scaler = MinMaxScaler(feature_range=(mi, ma))

        #scaler = StandardScaler()

        spat_data = data_spat.reshape(-1, data_spat.shape[-1])
        data_spat_new = scaler.fit_transform(spat_data).reshape(data_spat.shape)

        print('Dataset normalization Finished!')
        return data_spat_new

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6