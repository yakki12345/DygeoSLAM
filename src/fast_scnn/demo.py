import os
import argparse
import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
import cv2
import numpy as np
from utils.visualize import get_color_pallete
class Mask:
    def __init__(self):
        print('Initializing Mask CNN network...')
    # def replace(img,mask):
    #     r_img = img[:,:,0].copy()
    #     g_img = img[:,:,1].copy()
    #     b_img = img[:,:,2].copy()
    #     img = r_img * 256 * 256 + g_img * 256 + b_img
    #     src_c=0 * 256 * 256 + 0 * 256 + 142
    #     src_b=220 * 256 * 256 + 20 * 256 + 60
    #     mask[img==src_c]=1
    #     mask[img==src_b]=1
    #     return mask

    def GetDynSeg(image):
        # print("hello2")
        # print(torch.__version__)
        # print(torch.cuda.is_available())
        # print(torch.version.cuda)
        # print(torch.cuda.device_count())
        # print(type(image))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
        #WHEN IN GRAY
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #image=image.convert("RGB")
        # print("hello3")
        #image=np.array(image)
        sp=image.shape
        image= cv2.resize(image,(2048,1024)) 
        # print("hello4")
        # print("hello5")
        # x=np.transpose(image,[2,0,1])
        # y=x/255
        # mean, std = np.array([0.485, 0.456, 0.406]).reshape([3,1,1]), np.array([0.229, 0.224, 0.225]).reshape([3,1,1])
        # y2=(y-mean)/std
        # image=torch.from_numpy(y2)
        # image=image.float()
        # image=image.unsqueeze(0)
        image = transform(image).unsqueeze(0).to(device)
        # print(type(image))
        # print("hello6")
        # print(type(image))
        # image=image.unsqueeze(0).to(device)
        #transform=transforms.Compose([tosensor(image),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #image = transform(image).unsqueeze(0).to(device)
        #model = get_fast_scnn("citys", pretrained=True, root="./weights", map_cpu=False).to(device)
        # print("a2")
        #print(torch.load("/home/yakki/DynaSLAM_LK/src/fast_scnn/weights/fast_scnn_citys.pth", map_location='cpu'))
        model = get_fast_scnn("citys", pretrained=True, root="", map_cpu=False).to(device)
        # print('Finished loading model!')
        model.eval()
        # print("a1")
        # print(type(image))
        with torch.no_grad():
            # print("b")
            outputs=model(image)
        # print("a")
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, "citys")
        #outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
        mask=mask.convert("RGB")
        mask=mask.resize((sp[1],sp[0]))
        mask=np.array(mask)
        re=np.zeros((mask.shape[0],mask.shape[1]))
        r_img = mask[:,:,0].copy()
        g_img = mask[:,:,1].copy()
        b_img = mask[:,:,2].copy()
        img = r_img * 256 * 256 + g_img * 256 + b_img
        src_c=0 * 256 * 256 + 0 * 256 + 142
        src_b=220 * 256 * 256 + 20 * 256 + 60
        re[img==src_c]=1
        re[img==src_b]=1
        return re
    def say():
        print("hello")
        print("why i cannot")

    
	

    




    
	

    

