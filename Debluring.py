import time
from ASAPNet.options.train_options import TrainOptions
from multiprocessing import freeze_support
from ASAPNet.trainers.pix2pix_trainer import *
from ASAPNet.models.networks.generator import ASAPNetsGenerator
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from Project.ASAPNetTrial.ASAPNetMain.models1.losses import ContentLoss, DiscLoss


class Dataset(Dataset):
    def __init__(self, true_path):
        self.true_path = true_path
        self.all_path1 = sorted(os.listdir(self.true_path))

    def Normalize(self, inp):
        return inp / 255.
    
    def __getitem__(self, idx):
        ori_img = cv.imread(os.path.join(self.true_path, self.all_path1[idx])).astype(np.float32)
        ori_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)

            
        ori_img = cv.resize(ori_img, (256, 256))
        ori_img = torch.from_numpy(ori_img).unsqueeze(0)
        
        return self.Normalize(ori_img)

def test(model, valid_path):
    # opt.batchsize = 7
    dataset = Dataset(valid_path)
    dataloader = DataLoader(dataset, batch_size=1)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
    option = TrainOptions().parse()
    # model1 = ASAPNetsGenerator(option)
    model1 = model.netG
    print('Loading model ...\n')
    model1= model1.cuda()
    model1.load_state_dict(torch.load('blur_G.pth'))
    model1.eval()
    psnrMetric = []
    inv_normalize = T.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
                                std=[1/0.5, 1/0.5, 1/0.5] )
    
    k = 0
    for i, data in enumerate(dataloader):
        print('ith iteration: ', i)
        inputA = data['A'].cuda()
        inputB = data['B'].cuda()
        geny = model1(inputA)
        # print(geny.shape, inputB.shape)
        psnrMetric.append(PSNR(geny, inputB))
        # inv_tensor = inv_normalize(geny)
        for orr, im in zip(geny, inputA):
            # print(torch.min(im), torch.max(im))
            im = inv_normalize(im).detach().cpu().numpy().transpose(1, 2, 0)
            orr = inv_normalize(orr).detach().cpu().numpy().transpose(1, 2, 0)
            # print(np.min(orr), np.max(im))
            both = cv2.hconcat([orr*255, im*255])
            cv2.imwrite(f'pred/{str(k)}.png', both)
            k += 1
    print('Average PSNR: ', np.mean(psnrMetric))   

'''   
def train(data_loader, model):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    option = TrainOptions().parse()
    trainer = Pix2PixTrainer(option)
    tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
    discLoss, contentLoss = DiscLoss(opt, tensor), ContentLoss(nn.L1Loss())
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            # model.set_input(data)
            # model.optimize_parameters()
            inputA = data['A'].cuda()
            inputB = data['B'].cuda()

            generated, g_losses = trainer.pix2pix_model.netG.forward(inputA)
            # trainer.optimizer_D.zero_grad()
            # trainer.backward_D()
            loss_D = discLoss.get_loss(model.netD, inputA, generated.cuda(), inputB)
            loss_D.backward(retain_graph=True)
            trainer.optimizer_D.step()
            
            trainer.optimizer_G.zero_grad()
            # self.backward_G()
            loss_G_GAN = discLoss.get_g_loss(model.netD, inputA, generated)
        # Second, G(A) = B
            loss_G_Content = contentLoss.get_loss(generated, inputB) * opt.lambda_A

            loss_G = loss_G_GAN + loss_G_Content

            loss_G.backward()
            trainer.optimizer_G.step()
        
            if total_steps % opt.display_freq == 0:
                # results = model.get_current_visuals()
                # psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
                psnrMetric = PSNR(generated, inputB)
                print('PSNR on Train = %f' % psnrMetric)
                print(f'g losses: {sum(g_losses.detach().cpu().numpy()).mean()}')

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                # model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        if epoch % 2 == 0 :
            print(f'saving the model at the end of epoch {epoch}')
            # trainer.save('latest')
            trainer.save(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()
'''
def PSNR(img1, img2):
	img1 = img1.detach().cpu().numpy()
	img2 = img2.cpu().numpy()
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

if __name__ == '__main__':
    import argparse
    freeze_support()

    # python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_path", type=str, help='test data path')
    opt1 = parser.parse_args()
    opt1 = TrainOptions().parse()
    # opt.dataroot = 'D:\Photos\TrainingData\BlurredSharp\combined'
    # opt.dataroot = 'combined/newtest'
    # opt.learn_residual = True
    # opt.resize_or_crop = "crop"
    # opt.fineSize = 256
    # opt.gan_type = "gan"
    
    # opt.which_model_netG = "unet_256"

    # default = 5000
    # opt.save_latest_freq = 100

    # # default = 100
    # opt.print_freq = 20

    option = TrainOptions().parse()
    model = ASAPNetsGenerator(option)
    # train(data_loader, model)
    test(model, opt1.valid_path)
    