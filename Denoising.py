import numpy as np
import torch
import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from ASAPNet.models.networks.generator import ASAPNetsGenerator
from ASAPNet.options.train_options import TrainOptions, BaseOptions
import argparse
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as psnr
from ASAPNet.trainers.pix2pix_trainer import Pix2PixTrainer
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='val', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--choose", default='test', help='train or test')
parser.add_argument("--train_path", type=str, help='training data path')
parser.add_argument("--valid_path", type=str, help='test data path')
opt = parser.parse_args()


class DatasetNew(Dataset):
    def __init__(self, true_path):
        self.true_path = true_path
        # self.noise_path = noise_path
        self.all_path1 = sorted(os.listdir(self.true_path))
        # self.all_path2 = sorted(os.listdir(self.noise_path))

    def Normalize(self, inp):
        return inp / 255.
        # normalize = T.Normalize(mean=0.5, std=0.5)
        # return normalize(inp)
    
    def __getitem__(self, idx):
        ori_img = cv.imread(os.path.join(self.true_path, self.all_path1[idx])).astype(np.float32)
        ori_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
        # noise_img = cv.imread(os.path.join(self.noise_path, self.all_path2[idx])).astype(np.float32)
        # noise_img = cv.cvtColor(noise_img, cv.COLOR_BGR2GRAY)
        
        # if ori_img.shape == (321, 481):
        #     ori_img = ori_img.transpose(1, 0)
            # noise_img = noise_img.transpose(1, 0)
            
        # ori_img = cv.resize(ori_img, (512, 256))
        ori_img = cv.resize(ori_img, (256, 256))
        ori_img = torch.from_numpy(ori_img).unsqueeze(0)
        # noise_img = cv.resize(noise_img, (512, 256))
        # noise_img = torch.from_numpy(noise_img).unsqueeze(0)
        
        return self.Normalize(ori_img)#, self.Normalize(noise_img)
    
    def __len__(self):
        return len(self.all_path1)


    
def train(train_path):
    
    dataset_train = DatasetNew(train_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'
   
    print("# of training samples: %d\n" % int(len(dataset_train)))
    
    option = TrainOptions().parse()
    trainer = Pix2PixTrainer(option)
    noiseL = 25
    for epoch in range(50):

        # train
        print("*"*50)          
        # for i, (data, noisy_data) in enumerate(loader_train, 0):
        for i, data in enumerate(loader_train, 0):
            
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=noiseL/255.)
            imgn_train = img_train + noise
            
            #######################################################################################################################
            # if (option.D_steps_per_G == 0):
            #     generated, g_losses = trainer.run_generator_one_step(noisy_data - data, noisy_data)
            # elif (i % option.D_steps_per_G == 0):
            # #start = time.time()
            #     generated, g_losses = trainer.run_generator_one_step(noisy_data - data, noisy_data)
    

            # if (option.D_steps_per_G != 0):
            #     d_losses = trainer.run_discriminator_one_step(noisy_data - data, noisy_data)
            
            ########################################################################################################################
            generated, g_losses = trainer.run_generator_one_step(img_train, imgn_train)
            # d_losses = trainer.run_discriminator_one_step(img_train, imgn_train)
            
            if (i+1) % 9 == 0:
                # out_train = torch.clamp(imgn_train-generated.detach().cpu(), 0., 1.)
                # psnr_train = PSNR(noisy_data - generated.detach().cpu(), data)
                psnr_train = PSNR((generated.detach().cpu()), data)
                # print("[epoch %d][%d/%d] PSNR_train: %.4f G_loss: %.4f D_loss: %.4f" %
                #     (epoch+1, i+1, len(loader_train), psnr_train, g_losses, d_losses))
                print("[epoch %d][%d/%d] PSNR_train: %.4f G_loss: %.4f" %
                    (epoch+1, i+1, len(loader_train), psnr_train, g_losses))

        # ## the end of each epoch
        if epoch % 2 == 0 :
            print(f'saving the model at the end of epoch {epoch}')
            # trainer.save('latest')
            trainer.save(epoch)


def PSNR(img1, img2):
    # img1 = img1.detach().cpu().numpy()
    img1 = img1.numpy()
    img2 = img2.cpu().numpy()
	# mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def test(model, model_path, val_pure):
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'
    
    validation_data = DatasetNew(val_pure)
    loader = DataLoader(validation_data, batch_size=1)
    
    # Build model
    print('Loading model ...\n')
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
   
    psnr_test = 0
    cuda_timings = []
    # for idx, f in enumerate(files_source):
    for idx, ori in enumerate(loader):
        img_train = ori
        noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=25/255.)
        imgn_train = img_train + noise
        
        ori = ori.to(device)
        noi = imgn_train.to(device)
        sav_noisy = noi.squeeze().cpu().numpy()
        # print(sav_noisy.shape)
        # cv.imwrite(f'noisy/{str(idx)}.png', sav_noisy*255)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #GPU-WARM-UP
        with torch.no_grad(): # this can save much memory
            starter.record()
            extract_noise = model(noi)
            # Out = noi - extract_noise[0]
            Out = extract_noise[0]
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            cuda_timings.append(curr_time)
            pre = Out.cpu().squeeze().numpy()
            cv.imwrite(f'{str(idx)}.png', pre*255)
        psnr = batch_PSNR(Out, ori, 1.)
        psnr_test += psnr
        print("\nPSNR %f" % (psnr))
    psnr_test /= len(validation_data)
    print("\nPSNR on test data %f" % psnr_test)
    print("Inference time cuda: ", np.mean(cuda_timings))


if __name__ == '__main__':
    
    option = TrainOptions().parse()
    model = ASAPNetsGenerator(option)
    # model.apply(weights_init_kaiming)
    print('inside main111')
    if opt.choose == "train":
        # training
        train(opt.train_path)
    if opt.choose == 'test':
        valid_noise = "BSD/data/noisy_val"
        model_path = "denoising.pth"
        # testing
        test(model, model_path, opt.valid_path)
    