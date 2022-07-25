import shutil
import os
import cv2

scale=16
folder_orig="./samples"
dataset_list=["celeba"]
method_list=["BICUBIC", "SR_LRFR", "SR_RESNET", "SRGAN", "SR_WAVELET", "URWI"]
sr_images_root="C:/Users/Asus/PycharmProjects/AutoEncoder/LOGS"
for dataset in dataset_list:
    high_image_path=os.path.join(folder_orig, dataset)
    high_image_list=os.listdir(high_image_path)
    sr_folder=os.path.join(folder_orig, dataset+"_sr16")
    if not os.path.isdir(sr_folder):
        os.mkdir(sr_folder)
    for image_name in high_image_list:
        for method in method_list:
            method=method+str(scale)
            source_image=os.path.join(sr_images_root, method, "sr_images_"+dataset, "Images", image_name)
            dest_image=os.path.join(sr_folder, image_name[:-4]+"_"+method+".jpg")
            shutil.copyfile(source_image, dest_image)

