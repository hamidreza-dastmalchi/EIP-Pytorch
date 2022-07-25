import FNet
import cv2
import matplotlib.pyplot as plt
import FNet
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import shutil
import os
fnet = getattr(FNet, 'sface')()
fnet=fnet.cuda()
fnet.eval()

test_folder="C:/Users/Asus/PycharmProjects/datasets/aligned_lfw_all_in_one"
names_file="C:/Users/Asus/PycharmProjects/datasets/aligned_lfw_all_in_one/lfw_names.txt"
names=pd.read_csv(names_file, header=None)

image_name="./samples/12.png"
img=cv2.imread(image_name)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=cv2.resize(img, (128,128))
img=img.transpose(2,0,1)
img=torch.from_numpy(img).unsqueeze(0).cuda()
img=img.to(torch.float32)/255
img=2*(img-0.5)
img=img[:, :, 9:120, 17:112]
ref_feature=fnet(img).detach().cpu()

test_dataset=datasets.ImageFolder(test_folder, transform=transforms.ToTensor())
test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
test_features=[]
for count, data in enumerate(test_dataloader):
    print(count)
    test_img, _=data
    test_img=test_img.cuda()
    test_img=2*(test_img-0.5)
    test_img=test_img[:, :, 9:120, 17:112]
    with torch.no_grad():
        test_feature=fnet(test_img)
        test_features.append(test_feature.cpu())
test_features=torch.cat(test_features, dim=0)
feature_dist=((ref_feature-test_features).pow(2)).sum(dim=1)
dist_min, arg_min=feature_dist.unsqueeze(0).min(dim=1)
sel_name=names.iloc[arg_min.item()][0]

copy_fullname=os.path.join(test_folder,"Images", sel_name)
paste_fullname=os.path.join("./samples", sel_name )

shutil.copyfile(copy_fullname, paste_fullname)






