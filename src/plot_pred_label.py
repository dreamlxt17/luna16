# -*- coding:utf-8 -*-
from glob import glob
import numpy as np
import pickle
import gzip
from matplotlib import pyplot as plt

slice_path = '/home/didia/didia/data/1_1_1mm_slices_lung/subset8/'
label_path = '/home/didia/didia/data/1_1_1mm_slices_nodule/subset8/'
pre_list = glob('/home/didia/didia/luna16/result/tensor_predict_epoch82/*')
pre_list.sort()

threshold = 0.35


def pad_thres(im):
    # padding 成 (512x512)
    im_pad = np.pad(im, ((94,0), (0,94)), 'constant', constant_values=0)   # 水平方向上下，垂直方向左右
    im_pad[im_pad<threshold] = 0
    print im_pad.shape
    return im_pad


def main():

    for pre in pre_list[3:]:
        pre_name = pre.split('/')[-1].replace('.npy', '')
        print pre_name

        pred_tensor, spacing, origin = np.load(pre)     # 取预测data
        slice_list = glob(slice_path + pre_name + '*')  # 取同名的slice
        label_list = glob(label_path + pre_name + '*')  # 取同名的label
        print len(slice_list), len(label_list)

        for i in range(len(slice_list)):
            im1 = pickle.load(gzip.open(slice_list[i]))   # 原图
            im2 = pickle.load(gzip.open(label_list[i]))   # label
            im3 = pred_tensor[i]                          # 预测
            im3[im3<threshold] = 0
            im4 = pad_thres(im3)


            plt.subplot(221)
            plt.imshow(im1, cmap='gray')
            plt.subplot(222)
            plt.imshow(im2, cmap='gray')
            plt.subplot(223)
            plt.imshow(im3, cmap='gray')
            # plt.subplot(224)
            # plt.imshow(im4, cmap='gray')

            plt.show()
        break

main()

