# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import LoadDatasetFromFolder, DA_DatasetFromFolder, calculate_metrics
import numpy as np
import random
from model.network import CDNet
from train_options import parser
import itertools
from loss.losses import cross_entropy
import pandas as pd




args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seeds/efweresbn
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)
torch.cuda.empty_cache()
if __name__ == '__main__':
    mloss = 0
    nloss = 0
    # load data
    train_set = DA_DatasetFromFolder(args.hr1_train, args.hr2_train, args.lab_train, crop=False)
    val_set = LoadDatasetFromFolder(args, args.hr1_val, args.hr2_val, args.lab_val)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet(img_size = args.img_size).to(device, dtype=torch.float)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

    # set optimization
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= args.lr, betas=(0.9, 0.999))
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)

    results = {'train_loss':[],'f1':[],'recall':[],'precision':[],'oa':[],'iou':[]}

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'CD_loss':0, 'cross_loss': 0 }

        CDNet.train()
        for hr_img1, hr_img2, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label1 = label.to(device, dtype=torch.float)

            label = torch.argmax(label1, 1).unsqueeze(1).float()

            result1,  result2, result3,mmd_loss1,mmd_loss2,mmd_loss3 = CDNet(hr_img1, hr_img2)

            cross_loss = CDcriterionCD(result1, label)+CDcriterionCD(result2,label)+CDcriterionCD(result3,label)
            mmd_loss = (mmd_loss1 + mmd_loss2 + mmd_loss3)/3
            CD_loss = cross_loss + args.lf * mmd_loss
            CDNet.zero_grad()
            CD_loss.backward()
            optimizer.step()

            running_results['cross_loss'] += cross_loss.item() * args.batchsize
            running_results['CD_loss'] += CD_loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['CD_loss'] / running_results['batch_sizes'],))

        # eval
        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            # inter, unin = 0,0
            valing_results = {'batch_sizes': 0, 'IoU': 0}


            for hr_img1, hr_img2, label in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()

                cd_map,_,_,_,_,_= CDNet(hr_img1, hr_img2)

                CD_loss = CDcriterionCD(cd_map, label)

                cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()

                gt_value = (label > 0).float()
                prob = (cd_map > 0).float()

                prob = prob.cpu().detach().numpy()
                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)

                f1, recall, precision, oa, iou = calculate_metrics(result, gt_value)

                val_bar.set_description(
                    desc=f'F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, OA: {oa:.4f},IOU:{iou:.4f}')
            valing_results['F1']=f1
            valing_results['Recall'] = recall
            valing_results['Precision'] = precision
            valing_results['OA'] = oa
            valing_results['IOU'] = iou

        # save model parameters

        val_iou = valing_results['IoU']
        val_f1 = valing_results['F1']
        if val_iou > mloss or epoch==1 or val_f1 > nloss :
            mloss = val_iou
            nloss = val_f1
            torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_%d.pth' % (epoch))

        results['train_loss'].append(running_results['CD_loss'] / running_results['batch_sizes'])
        results['f1'].append(valing_results['F1'])
        results['recall'].append(valing_results['Recall'])
        results['precision'].append(valing_results['Precision'])
        results['oa'].append(valing_results['OA'])
        results['iou'].append(valing_results['IOU'])

        if epoch % 1 == 0 :
            data_frame = pd.DataFrame(
                data={'train_loss': results['train_loss'],
                      'F1': results['f1'],
                      'Recall:':results['recall'],
                      'Precision':results['precision'],
                      'OA':results['oa'],
                      'IOU':results['iou']},
                index=range(1, epoch + 1))
            data_frame.to_csv("./statistic/statistic.csv", index_label='Epoch')
