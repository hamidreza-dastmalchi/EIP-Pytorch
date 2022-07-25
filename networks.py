import torch.nn as nn
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms



class EdgeBlock(nn.Module):
    def __init__(self, InCh, KernelAvg):
        super(EdgeBlock, self).__init__()
        self.AvgLayer = nn.AvgPool2d(kernel_size=KernelAvg, stride=1, padding=(KernelAvg-1)//2)
        self.bottleneck = nn.Conv2d(in_channels=InCh, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_avg = self.AvgLayer(x)
        edge_feature = self.relu(x-x_avg)
        edge = self.relu(self.bottleneck(edge_feature))
        return edge_feature, edge


class Res_EPI(nn.Module):
    def __init__(self, InCh, OutCh, KernelAvg):
        super(Res_EPI, self).__init__()
        self.KernelAvg = KernelAvg
        conv1 = nn.Conv2d(in_channels=InCh, out_channels=InCh, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(in_channels=InCh, out_channels=OutCh, kernel_size=3, stride=1, padding=1)
        self.ResBlock=nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        if KernelAvg != None:
            self.EdgeBlock = EdgeBlock(InCh//2, KernelAvg)

    def forward(self,x):
        if self.KernelAvg!=None:
            edge_feature, edge = self.EdgeBlock(x)
            x_in_res = torch.cat((edge_feature, x), dim=1)
        else:
            x_in_res = x
            edge = []
        output = self.ResBlock(x_in_res) + x
        return output, edge



class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.conv_head = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1)
        feature_num = 512
        self.res1 = Res_EPI(512, 512, None)
        avg_kernel=[5, 7]
        for i in range(2):
            self.add_module("res{}".format(i+2), Res_EPI(feature_num, feature_num//2, avg_kernel[i]))
            feature_num = feature_num//2
        feature_num = 512
        for i in range(3):
            self.add_module("ConvTr{}".format(i+1), nn.Sequential(
                nn.ConvTranspose2d(in_channels=feature_num, out_channels=feature_num//2, kernel_size=4, padding=1, stride=2),
                nn.ReLU()))
            feature_num = feature_num//2
        self.EdgeBlock_End = EdgeBlock(64, 9)
        self.conv_tail = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_head(x)
        x, _ = self.res1(x)
        edges = []
        for i in range(2):
            x = self.__getattr__("ConvTr{}".format(i+1))(x)
            x, edge = self.__getattr__("res{}".format(i+2))(x)
            edge = torch.clamp(edge, 0, 1)
            edges.append(edge)
        x = self.ConvTr3(x)
        edge_feature, edge = self.EdgeBlock_End(x)
        edge = torch.clamp(edge, 0, 1)
        edges.append(edge)
        x = torch.cat((edge_feature, x), dim=1)
        out = self.conv_tail(x)
        out = torch.clamp(out, 0, 1)
        return out, edges


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        in_channels = [3, 128, 128, 256, 256, 256, 512]
        out_channels = [128, 128, 256, 256, 256, 512, 512]
        strides = [1, 2, 1, 2, 1, 2, 1]
        paddings = [1, 0, 1, 0, 1, 0, 1]
        self.NumConv = len(in_channels)
        for i in range(len(in_channels)):
            if i!=len(in_channels):
                self.add_module("Conv{}".format(i+1),
                            nn.Sequential(nn.Conv2d(in_channels=in_channels[i], out_channels=out_channels[i],
                                                    kernel_size=3, stride=strides[i], padding=paddings[i]),
                                          nn.LeakyReLU(0.2)))
            else:
                self.add_module("Conv{}".format(i+1),
                                nn.Conv2d(in_channels=in_channels[i], out_channels=out_channels[i],
                                                    kernel_size=3, stride=strides[i], padding=paddings[i]))
        self.linear1 = nn.Sequential(nn.Linear(15*15*512, 512), nn.LeakyReLU(0.2))
        self.linear2 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        for i in range(self.NumConv):
            x = self.__getattr__("Conv{}".format(i+1))(x)
        x = x.reshape(20, -1)
        x = self.linear1(x)
        out = self.linear2(x)
        return out

def LowEdgeGeneration(images_high):
    images_high = images_high.cpu().numpy().transpose(0, 2, 3, 1)
    nim, nrow, ncol, nch = images_high.shape
    image_low = np.empty((nim, nrow//8, ncol//8, nch), dtype=images_high.dtype)
    edges_128 =  np.empty((nim, nrow, ncol), dtype=images_high.dtype)
    edges_64 =  np.empty((nim, nrow//2, ncol//2), dtype=images_high.dtype)
    edges_32 =  np.empty((nim, nrow//4, ncol//4), dtype=images_high.dtype)
    for i in range(images_high.shape[0]):
        img = images_high[i]
        img_lr_64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        img_lr_32 = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
        img_lr_16 = cv2.resize(img, (16, 16), interpolation=cv2.INTER_LINEAR)
        edge_128 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge_128 = cv2.Canny((255*edge_128).astype(np.uint8), 50, 255)
        edge_128 = edge_128.astype(np.float32)/255
        edge_64 = cv2.cvtColor(img_lr_64, cv2.COLOR_RGB2GRAY)
        edge_64 = cv2.Canny((255*edge_64).astype(np.uint8), 50, 255)
        edge_64 = edge_64.astype(np.float32) / 255
        edge_32 = cv2.cvtColor(img_lr_32, cv2.COLOR_RGB2GRAY)
        edge_32 = cv2.Canny((255*edge_32).astype(np.uint8), 50, 255)
        edge_32 = edge_32.astype(np.float32) / 255
        image_low[i] = img_lr_16
        edges_128[i] = edge_128
        edges_64[i] = edge_64
        edges_32[i] = edge_32
    image_low = torch.from_numpy(image_low.transpose(0,3,1,2))
    edges_128 = torch.from_numpy(edges_128).unsqueeze(1)
    edges_64 = torch.from_numpy(edges_64).unsqueeze(1)
    edges_32 = torch.from_numpy(edges_32).unsqueeze(1)
    return image_low, edges_128, edges_64, edges_32


def RGB2YUV(rgb_batch):
    yuv_batch = torch.empty_like(rgb_batch)
    red_batch = rgb_batch[:,0,:,:]
    green_batch = rgb_batch[:, 1, :, :]
    blue_batch = rgb_batch[:, 2, :, :]
    y_batch = 0.299*red_batch + 0.587*green_batch + 0.114*blue_batch
    u_batch = -0.14713*red_batch + -0.28886*green_batch + 0.436*blue_batch
    v_batch = 0.615*red_batch + -0.51499*green_batch + -0.10001*blue_batch
    yuv_batch[:,0,:,:] =  y_batch
    yuv_batch[:,1,:,:] =  u_batch
    yuv_batch[:,2,:,:] =  v_batch
    return yuv_batch

"""
def FaceBoxExtract(img_batch):
    PilTrf = transforms.ToPILImage()
    img_bath = img_batch.cpu()
    BoundingBoxes = np.empty((img_batch.size()[0], 5))
    for i in range(img_bath.size()[0]):
        pil_image = PilTrf(img_bath[i])
        bounding_box, landmarks = detect_faces(pil_image)
        if len(bounding_box)==1:
            bounding_box = bounding_box[0]
        if bounding_box==[] or bounding_box[0]>45:
            print(bounding_box)
            bounding_box = np.array([20, 20, 115, 130, 1], dtype=BoundingBoxes.dtype)
        BoundingBoxes[i] = bounding_box
    return BoundingBoxes

def FaceCrop(img_batch, BoundingBoxes):
    nim, nch, nrow, ncol = img_batch.size()
    face_batch = torch.empty(nim, nch, 160, 160)
    for i in range(nim):
        bound_box = BoundingBoxes[i]
        img = img_batch[i]
        face = img[:,bound_box[1]: bound_box[3], bound_box[0]: bound_box[2]]
        face = upsample_box(face.unsqueeze(0)).squeeze(0)
        face_batch[i] = face
    return face_batch
"""





