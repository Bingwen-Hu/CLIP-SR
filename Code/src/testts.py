import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
import math


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_models_opt,save_models_opt,save_models,load_npz,params_count
from lib.perpare import prepare_dataloaders,prepare_models
# from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets import get_fix_data
from lib.datasets import prepare_data
from tqdm import tqdm, trange


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/birds.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='/opt/data/private/carr/code/saved_models/bird/GALIP_nf64_normal_bird_256_2025_03_12_08_15_52',
                        help='the model for training')

    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=210,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=1 ,
                        help='batch size')
    parser.add_argument('--train', type=str, default='Fasle',
                        help='if train model')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--scaler', default=4, type=int,
                        help='2,3,4,8')
    parser.add_argument('--random_sample', action='store_true',default=True,
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args):

    #--------------------------------------图片保存路径------------------------------------
    time_stamp = get_time_stamp()
    #2023_03_13_19_55_47
    name='TEST'
    stamp = '_'.join([str(name),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    #GALIP_nf64_normal_bird_128_2023_03_13_19_57_41

    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'test', stamp)))

    #G:\workspaces\GALIP-main\GALIP-main\code\imgs/bird\train\GALIP_nf64_normal_bird_256_2023_03_15_09_30_59
    #------------------------------------------------------------------------------------------------------
    mkdir_p(args.img_save_dir)
    #============================================ prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)

    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)

    print('**************G_paras: ',params_count(netG))
    print('**************D_paras: ',params_count(netD)+params_count(netC))
    # fixed_img,LR, fixed_sent, fixed_words, fixed_z = get_fix_data(train_dl, valid_dl, text_encoder, args)

    # ############################prepare optimizer,设置学习率等=================================
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=args.lr_d, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    start_epoch = 1
    # ==================================================load from checkpoint===================================

    if args.state_epoch!=1:
        start_epoch = args.state_epoch + 1
        path = osp.join(args.pretrained_model_path, 'state_epoch_%03d.pth'%(args.state_epoch))
        print("55555\n")
        print(path)
        netG, netD, netC, optimizerG, optimizerD = load_models_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)

        #G:\workspaces\GALIP-main\GALIP-main\code\logs/bird\train\GALIP_nf64_normal_bird_128_2023_03_13_20_41_16\args.yaml

    # ===================================================================Start training

    for epoch in range(start_epoch, args.max_epoch, 1):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        args.current_epoch = epoch

        sample(valid_dl,netG,args.img_save_dir,text_encoder,args)
        print("-----------------test-----finish--------------")
        exit(0)
        torch.cuda.empty_cache()

from skimage.metrics import structural_similarity as ssim

from niqe import niqe
import numpy as np

def safe_niqe(img):
    try:
        # 确保传入的图像尺寸和范围符合要求
        # 这里假设 niqe 函数需要一个 2D 图像（灰度图像），而不是 RGB 图像
        if len(img.shape) == 3 and img.shape[2] == 3:  # 如果是 RGB 图像
            img = np.mean(img, axis=2)  # 转为灰度图像

        return niqe(img)  # 计算 NIQE
    except ValueError as e:
        print(f"Error in NIQE calculation: {e}")
        return float('nan')  # 返回 NaN 以避免计算中断

def calculate_pi(sr_img, real_img):
    # PI 是结合 SSIM 和 PSNR 的指标，通常可以通过某种方式加权计算
    # 这里我们使用一个简单的加权平均作为示例：
    psnr_value = calculate_psnr(sr_img, real_img)
    min_size = min(sr_img.shape[:2])  # 获取最小边长
    win_size = min(7, min_size)  # 确保 win_size <= 7，并且 <= 最小边长
    if win_size >= 3:  # 确保 win_size 至少为 3
        ssim_value = ssim(sr_img, real_img, channel_axis=-1, data_range=255, win_size=win_size)
    else:
        ssim_value = 0  # 图像太小，无法计算 SSIM

    # PI = w1 * PSNR + w2 * SSIM
    # 这里 w1 和 w2 是你可以调整的权重
    w1, w2 = 0.5, 0.5
    return w1 * psnr_value + w2 * ssim_value
def sample(dataloader, netG, img_save_dir, text_encoder, args):
    device = args.device
    loop = tqdm(total=len(dataloader))
    netG.eval()

    psnr_values = []
    ssim_values = []
    niqe_values = []
    pi_values = []

    for step, data in enumerate(dataloader, 0):
        real, LR, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)

        with torch.no_grad():
            SR = generate_samples(LR, sent_emb, netG).to(device)

        # 转换 tensor 为 numpy 图像
        real_img = tensor2img(real)  # 原始高分辨率图像
        sr_img = tensor2img(SR)  # 生成的超分辨率图像

        # 计算 PSNR 和 SSIM
        psnr_value = calculate_psnr(sr_img, real_img)
        min_size = min(sr_img.shape[:2])  # 获取最小边长
        win_size = min(7, min_size)  # 确保 win_size <= 7，并且 <= 最小边长
        if win_size >= 3:  # 确保 win_size 至少为 3
            ssim_value = ssim(sr_img, real_img, channel_axis=-1, data_range=255, win_size=win_size)
        else:
            ssim_value = 0  # 图像太小，无法计算 SSIM

        # 计算 NIQE
        niqe_value = safe_niqe(sr_img)

        # 计算 PI
        pi_value = calculate_pi(sr_img, real_img)

        # 添加结果到列表
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        niqe_values.append(niqe_value)
        pi_values.append(pi_value)

        # 保存图像
        img_name = f"{step}.jpg"
        img_save_path = osp.join(img_save_dir, img_name)
        img_save_path_HR = osp.join("/opt/data/private/carr/dataset/wu", img_name)

        vutils.save_image(SR.data, img_save_path, nrow=1, value_range=(-1, 1), normalize=True)
        vutils.save_image(real.data, img_save_path_HR, nrow=1, value_range=(-1, 1), normalize=True)

        loop.update(1)
        loop.set_description(f'Testing [{step}/{len(dataloader)}]')

    loop.close()

    # 计算平均 PSNR, SSIM, NIQE 和 PI
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_niqe = np.mean(niqe_values)
    avg_pi = np.mean(pi_values)

    print(f"\n=== Evaluation Results ===")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average NIQE: {avg_niqe:.4f}")
    print(f"Average PI: {avg_pi:.4f}")
    print(f"--------------------------------------Test Finished-------------------------------------------")


def generate_samples(LR, caption, model):
    with torch.no_grad():
        SR = model(LR, caption, eval=True)
    return SR
def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).detach().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

