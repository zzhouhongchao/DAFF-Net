# python imports
import os
import glob
import warnings
import sys
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
import argparse
import torch.nn.functional as F
# internal imports
# import tensorboard
# from torch.utils.tensorboard import SummaryWriter
import losses
from generators import DatasetOASIS3_seg

from Networks import final_semi_net



def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.Log_dir):
        os.makedirs(args.Log_dir)
    # if not os.path.exists(args.test_result):
    #     os.makedirs(args.test_result)
            
def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    # img = sitk.GetImageFromArray(img)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.test_result, name))
    
def train(args):
    make_dirs()
    # writer = SummaryWriter(args.Log_dir)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # f_img = sitk.ReadImage("../data/LPBA/test/S01.delineation.skullstripped.nii.gz")
    log_name = 'OASIS3_semi'
    print("log_name: ", log_name)
    f = open(os.path.join(args.Log_dir, log_name + ".txt"), "a")  
    
     
    model = final_semi_net.Net(for_train=True,use_checkpoint=True)
    model.to(device)
    # model.load_state_dict(torch.load('model/experiments/semi_woatt_L2_KL_10/410.pth'))
    model.train()
    
    opt = Adam(model.parameters(), lr=args.lr)
    DICELOSS = losses.FocalDice().loss
    
    train_files = glob.glob(os.path.join(args.train_dir,'train', '*.nii.gz'))
    DS = DatasetOASIS3_seg(files=train_files) 
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
   
    for i in range (1,args.epoch + 1 ):
        for data in DL:
            gray,seg,name = data
            
            input_moving = gray[0:1]
            input_mov_seg = seg[0:1]
            input_moving = input_moving.to(device).float()
            input_mov_seg = input_mov_seg.to(device).float()
            
            input_fixed = gray[1:2]
            input_fix_seg = seg[1:2]
            input_fixed = input_fixed.to(device).float()
            input_fix_seg = input_fix_seg.to(device).float()
            
            flow_1,warped1,warped_mov_seg,warpeed_pre_mov_seg,pre_mov_seg,pre_fix_seg= model(input_moving,input_fixed,input_mov_seg)

            sim_loss1 =losses.ncc_loss(warped1,input_fixed)
            grad_loss1 = losses.gradient_loss(flow_1)
            NJ_loss1 = losses.NJ_loss()(flow_1.permute(0, 2, 3, 4, 1))

            dice1 = DICELOSS(input_mov_seg,pre_mov_seg)
            dice2 = DICELOSS(input_fix_seg,pre_fix_seg)
            dice3 = DICELOSS(input_fix_seg,warped_mov_seg)
            dice4 = DICELOSS(input_fix_seg,warpeed_pre_mov_seg)
            dice5 = DICELOSS(pre_fix_seg,warpeed_pre_mov_seg)
           
            dice_f = dice1 + dice2
            dice_r = dice3 + dice4 + dice5 
            loss = sim_loss1 + grad_loss1 + 0.00001 * NJ_loss1 + 0.5 * dice_f + dice_r
            
            print("i: %d name: %s loss: %f sim: %f L2: %f  dice: %f   njd: %f "
                  % (i, name, loss.item(), sim_loss1.item(), grad_loss1.item(), dice_r.item(),NJ_loss1.item()),flush=True)                                                                                                         
            print("%d, %s, %f, %f, %f,%f,%f"
                  % (i, name, loss.item(), sim_loss1.item(), grad_loss1.item(), dice_r.item(),NJ_loss1.item()),file=f)
           
            opt.zero_grad() 
            loss.backward() 
            opt.step()
            
        if (i % 10 == 0):
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(model.state_dict(), save_file_name)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
    parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
    parser.add_argument("--epoch", type=int, help="number of iterations",
                    dest="epoch", default=150)
    parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=2)
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)
    parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="../data/OASIS3/")
    parser.add_argument("--model_dir", type=str, help="data folder with training vols",
                    dest="model_dir", default="model/newexperiments/OASIS3_semi")
    parser.add_argument("--Log_dir", type=str, help="data folder with training vols",
                    dest="Log_dir", default="logs/newexperiments/OASIS3_semi")
    args = parser.parse_args()
    train(args)
