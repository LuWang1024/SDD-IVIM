# -*- coding: UTF-8 -*-
'''
Created on Wed Oct 9 20:15:00 2019

@author: Qinqin
'''
import os
import argparse
import numpy as np
import scipy.io as matio

from UNet import Inference

from tools.data_loader import get_loader
from tools.evaluation import *

def test(config):
    #-----选择GPU-----#
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    #-----使每次生成的随机数相同-----#
    np.random.seed(1)
    torch.manual_seed(1)

    # -----地址-----#
    model_dir = os.path.join(config.model_path, config.name+'/'+ config.name+ '_epoch_' +config.model_num + '.pth')
    if not os.path.exists(model_dir):
        print('Model not found, please check you path to model')
        os._exit(0)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    #-----读取数据-----#
    test_batch = get_loader(config.data_dir, config, crop_key=False, num_workers=1, shuffle=False, mode=config.test_dir)

    #-----模型-----#
    net = Inference(config.INPUT_C,config.OUTPUT_C,config.FILTERS)

    if torch.cuda.is_available():
        net.cuda()

    #-----载入模型参数-----#
    torch.load(model_dir)
    net.load_state_dict(torch.load(model_dir))
    print('Model parameters loaded!')

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ********************************************test*****************************************************#
    net.eval()
    for i,(images, GT) in enumerate(test_batch):
        images = images.type(torch.FloatTensor)
        GT = GT.type(torch.FloatTensor)

        images = images.to(device)

        SR = net(images)  # forward

        if i == 0:
            X_test = images.permute(0, 2, 3, 1).cpu().detach().numpy()
            Y_test = GT.permute(0, 2, 3, 1).cpu().detach().numpy()
            OUT_test = SR.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            X_test = np.concatenate((images.permute(0, 2, 3, 1).cpu().detach().numpy(),X_test),axis=0)
            Y_test = np.concatenate((GT.permute(0, 2, 3, 1).cpu().detach().numpy(),Y_test),axis=0)
            OUT_test = np.concatenate((SR.permute(0, 2, 3, 1).cpu().detach().numpy(),OUT_test),axis=0)

    #-----保存为mat文件-----#
    print('.' * 30)
    # print('X_test:', X_test.shape)
    # print('Y_test:', Y_test.shape)
    print('OUT_test:', OUT_test.shape)
    print('.' * 30)
    matio.savemat(
        os.path.join(config.result_path, config.name + '_result_' + config.test_dir + '.mat'),
        {
            # 'input': X_test,
            #'label': Y_test,
            'output': OUT_test
        })
    print('Save result in ',config.name + '_result_' + config.test_dir + '.mat')
    print('.' * 30)
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--GPU_NUM', type=str, default='4')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=128)
    parser.add_argument('--INPUT_W', type=int, default=128)
    parser.add_argument('--INPUT_C', type=int, default=10)
    parser.add_argument('--OUTPUT_C', type=int, default=1)
    parser.add_argument('--LABEL_C', type=int, default=3)
    parser.add_argument('--DATA_C', type=int, default=13)
    parser.add_argument('--FILTERS', type=int, default=64)

    # test hyper-parameters
    parser.add_argument('--BATCH_SIZE', type=int, default=1)
    parser.add_argument('--CROP_SIZE', type=int, default=96)

    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--result_path', type=str, default='./test_result/')
    parser.add_argument('--test_dir', type=str, default='')

    config = parser.parse_args()

    config.model_num = '2000'
    config.test_dir = 'brain'
    config.name = 'D'
    test(config)