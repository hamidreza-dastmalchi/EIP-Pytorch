from facenet_pytorch import InceptionResnetV1
from networks import generator, discriminator
from networks import LowEdgeGeneration, RGB2YUV
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import torch
import random
import os





number_epochs=200
batch_size=20
lr=1e-4
scale=8
gen_betas=(0.9, 0.999)
disc_betas=(0.5, 0.9)


seed=123
num_iter_show=50
id_avg=32**2+64**2+128**2
#train_path="C:/Users/Asus/PycharmProjects/IDAGAN/CelebATrain"
train_path="C:/Users/Asus/PycharmProjects/WaveletSRNet-master/data"
test_path="C:/Users/Asus/PycharmProjects/AutoEncoder/LOGS/Original_Images_Test"
CheckpointsSR1="./CheckSR1"
if not (os.path.isdir(CheckpointsSR1)):
    os.mkdir(CheckpointsSR1)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
upsample_box = nn.Upsample(size=(160, 160), mode="bilinear", align_corners=True)
#########################################################################
train_dataset=datasets.ImageFolder(train_path, transform=transforms.ToTensor())
train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,  num_workers=2)

test_dataset=datasets.ImageFolder(test_path, transform=transforms.ToTensor())
test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

writer=SummaryWriter("./logs")
test_dataloader_iter=iter(test_dataloader)
num_batch=len(train_dataset)//batch_size
upsample=nn.Upsample(128, mode="bilinear", align_corners=True)

#########################################################################
high_image_eval, _ = next(test_dataloader_iter)
high_image_eval=high_image_eval[0:16, :, : , :]
high_image_eval_grid=make_grid(high_image_eval, nrow=4, padding=4)
low_image_eval, edge_128_eval, edge_64_eval, edge_32_eval=LowEdgeGeneration(high_image_eval)
low_image_eval_up=upsample(low_image_eval)
low_image_eval_up_grid=make_grid(low_image_eval_up, nrow=4, padding=4)
low_image_eval=low_image_eval.cuda()
writer.add_image("Original Eval Images", high_image_eval_grid, 0)
writer.add_image("Input Eval Images", low_image_eval_up_grid, 0)


#########################################################################
gen_net=generator().cuda()
disc_net=discriminator().cuda()
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
#gen_net.load_state_dict(torch.load(os.path.join(checkpoints,"gen_net90")))
#gen_net.load_state_dict(torch.load(os.path.join(checkpoints,"gen_net50")))
#gen_net.apply(weights_init)
#disc_net.apply(weights_init)
#########################################################################
gen_optim=optim.Adam(gen_net.parameters(), lr=lr, betas=gen_betas)
gen_schedul=lr_scheduler.StepLR(gen_optim, 50, .5)
disc_optim=optim.Adam(disc_net.parameters(), lr=lr, betas=disc_betas)
disc_schedul=lr_scheduler.StepLR(disc_optim, 50, 0.5)
BCE_Loss=nn.BCELoss().cuda()
MSE_Loss=nn.MSELoss(reduction="mean").cuda()
softmax = nn.Softmax(dim=1).cuda()
#########################################################################
def main():
    str_time=time.time()
    counter=0
    num=0
    for ep in range(number_epochs):
        gamma = 0.3
        beta = 0.001
        alpha = 0.001
        if ep%10==9:
            gen_net.eval()
            disc_net.eval()
            torch.save(gen_net.state_dict(), os.path.join(CheckpointsSR1,"EipGenNet_{}".format(ep+1)))
            torch.save(disc_net.state_dict(), os.path.join(CheckpointsSR1, "EipDiscNet_{}".format(ep + 1)))
        gen_adv_loss = 0
        gen_id_loss = 0
        gen_mse_loss = 0
        gen_edge_loss = 0
        gen_yuv_loss = 0
        gen_total_loss = 0
        disc_real_loss = 0
        disc_fake_loss = 0
        disc_total_loss = 0
        gen_net.train()
        disc_net.train()
        for count, data in enumerate(train_dataloader):
            high_image , _ = data
            low_image, edge_128, edge_64, edge_32 = LowEdgeGeneration(high_image)
            low_image = low_image.cuda()
            high_edges=[edge_32.cuda(), edge_64.cuda(), edge_128.cuda()]
            high_image = high_image.cuda()
            high_face = high_image[:, :, 22:135, 18:105]
            high_face = upsample_box(high_face)
            high_image_yuv = RGB2YUV(high_image)
            #####################################################################################################
            if ep>=2:
                with torch.no_grad():
                    sr_image, sr_edges = gen_net(low_image)
                d_fake = disc_net(sr_image)
                d_real = disc_net(high_image)
                dloss_fake = BCE_Loss(d_fake, 0.1*torch.ones_like(d_fake).cuda())
                dloss_real = BCE_Loss(d_real, 0.9*torch.ones_like(d_real).cuda())
                dloss = dloss_real + dloss_fake
                disc_total_loss += dloss.item()
                disc_real_loss += dloss_real.item()
                disc_fake_loss += dloss_fake.item()
                disc_optim.zero_grad()
                dloss.backward()
                disc_optim.step()
            #####################################################################################################
            sr_image, sr_edges = gen_net(low_image)
            sr_image_yuv = RGB2YUV(sr_image)
            if ep>=4:
                d_fake_g = disc_net(sr_image)
                gadv_loss = BCE_Loss(d_fake_g, torch.ones_like(d_fake_g).cuda())
            else:
                gadv_loss = torch.zeros(1, dtype=torch.float32).cuda()
            gmse_loss = MSE_Loss(sr_image, high_image)
            gyuv_loss = MSE_Loss(sr_image_yuv, high_image_yuv)
            gedge_loss = 0
            for i in range(len(sr_edges)):
                gedge_loss += MSE_Loss(high_edges[i], sr_edges[i])*(32**2)*(2**(2*i))
            gedge_loss = gedge_loss/id_avg
            sr_face = sr_image[:, :, 22:135, 18:105]
            sr_face = upsample_box(sr_face)
            sr_identity = softmax(10*resnet(sr_face))
            high_identity = softmax(10*resnet(high_face))
            identity_mean = (sr_identity+high_identity)/2
            kl_sr = 0.5*sr_identity*torch.log2(sr_identity/identity_mean)
            kl_hr = 0.5*high_identity*torch.log2(high_identity/identity_mean)
            gjs_loss = (kl_sr+kl_hr).sum(dim=1).mean()
            gen_optim.zero_grad()
            disc_optim.zero_grad()
            gen_loss =  gmse_loss + gamma * gedge_loss +  gyuv_loss + alpha * gjs_loss + beta * gadv_loss
            gen_total_loss += gen_loss.item()
            gen_adv_loss += gadv_loss.item()
            gen_id_loss += gjs_loss.item()
            gen_mse_loss += gmse_loss.item()
            gen_yuv_loss += gyuv_loss.item()
            gen_edge_loss += gedge_loss.item()
            gen_loss.backward()
            gen_optim.step()
            #####################################################################################################
            if count % num_iter_show == num_iter_show - 1:
                end_time = time.time()
                gen_net.eval()
                dur_time=end_time-str_time
                with torch.no_grad():
                    sr_image_eval, sr_edges_eval = gen_net(low_image_eval)
                    for indw in range(len(sr_edges_eval)):
                        sr_edge_eval = sr_edges_eval[indw].cpu()
                        sr_edge_eval_grid = make_grid(sr_edge_eval, nrow=4, padding=4)
                        writer.add_image("super-resolved edge images in scale{}" + str(128/(2**(2-indw))), sr_edge_eval_grid, counter)
                    sr_image_eval = sr_image_eval.cpu()
                    eval_sr_grid = make_grid(sr_image_eval, nrow=4, padding=4)
                    writer.add_image("Images Super-Resolved by EIP network", eval_sr_grid, counter)
                writer.add_scalar("generator adversarial loss", gen_adv_loss/num_iter_show, counter)
                writer.add_scalar("generator identity loss", gen_id_loss / num_iter_show, counter)
                writer.add_scalar("generator rgb mse loss", gen_mse_loss / num_iter_show, counter)
                writer.add_scalar("generator yuv mse loss", gen_yuv_loss / num_iter_show, counter)
                writer.add_scalar("generator edge loss", gen_edge_loss / num_iter_show, counter)
                writer.add_scalar("discriminator real loss", disc_real_loss/num_iter_show, counter)
                writer.add_scalar("discriminator fake loss", disc_fake_loss / num_iter_show, counter)
                writer.add_scalar("discriminator total loss", disc_total_loss / num_iter_show, counter)
                print("epoch {0:03d}/{1:03d} \t iter {2:04d}/{3:04d} \t gen adv loss {4:0.4f} \t gen id loss {5:0.4f} "
                      "\t gen rgb mse loss {6:0.4f} \t gen yuv mse loss {7:0.4f} \t gen edge loss {8:0.4f} \t dis adv loss {9:0.4f} \t time: {10:0.02f}".
                    format(ep+1, number_epochs, count+1, num_batch, gen_adv_loss/num_iter_show, gen_id_loss/num_iter_show,
                           gen_mse_loss/num_iter_show, gen_yuv_loss/num_iter_show, gen_edge_loss/num_iter_show, disc_total_loss / num_iter_show, dur_time))
                gen_adv_loss = 0
                gen_id_loss = 0
                gen_mse_loss = 0
                gen_yuv_loss = 0
                gen_edge_loss = 0
                gen_total_loss = 0
                disc_real_loss = 0
                disc_fake_loss = 0
                disc_total_loss = 0
                counter += 1
                gen_net.train()
                str_time = time.time()
        disc_schedul.step()
        gen_schedul.step()


if __name__=="__main__":
    main()



