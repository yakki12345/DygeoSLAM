import os
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()


def demo():
    print('Initializing Mask CNN network...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.input_pic).convert('RGB')
    image=np.asarray(image)
    sp=image.shape
    image= cv2.resize(image,(2048,1024)) 
    print(type(image))
    image = transform(image).unsqueeze(0).to(device)
    print(type(image))
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    mask=mask.convert("RGB")
    mask=mask.resize((sp[1],sp[0]))
    mask=np.array(mask)
    re=np.zeros((mask.shape[0],mask.shape[1]))
    for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if((mask[i][j][0]==220)and(mask[i][j][1]==20)and(mask[i][j][2]==60)):
                    re[i][j]=255
    	        #if(((mask[i][j][0]==0)and(mask[i][j][1]==0)and(mask[i][j][2]==142))or((mask[i][j][0]==220)and(mask[i][j][1]==20)and(mask[i][j][2]==60))):
    	            # re[i][j]=1
    # r_img = mask[:,:,0].copy()
    # g_img = mask[:,:,1].copy()
    # b_img = mask[:,:,2].copy()
    # img = r_img * 256 * 256 + g_img * 256 + b_img
    # src_c=0 * 256 * 256 + 0 * 256 + 142
    # src_b=220 * 256 * 256 + 20 * 256 + 60
    # re[img==src_c]=255#car
    # re[img==src_b]=255#??
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    cv2.imwrite('/home/yakki/DynaSLAM_LK/src/fast_scnn/34.png',re)
    #re.save('/home/yakki/DynaSLAM_LK/src/fast_scnn/34.png')


if __name__ == '__main__':
    demo()
