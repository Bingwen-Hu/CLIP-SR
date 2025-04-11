import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.insert(0, '../')
from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets import get_fix_data
from PIL import Image
import numpy as np

from lib.utils import load_model_weights,mkdir_p
from models.GALIP import NetG, CLIP_TXT_ENCODER,CLIP_IMG_ENCODER
device = 'cpu' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
netG = NetG(64, 512, 512, 256, 3, False, clip_model).to(device)
path = '../saved_models/pretrained/pre_cc12m.pth'
path = '/opt/data/private/workspace/GALIP-main/code/saved_models/bird/GALIP_nf64_normal_bird_256_2023_04_25_00_39_37/state_epoch_440.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)


batch_size = 1
noise = torch.randn((batch_size, 100)).to(device)
norm = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.unsqueeze(0),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
            ])
image_transform = transforms.Compose([
            transforms.Resize(int(256 * 76 / 64)),
            transforms.CenterCrop(256),
            ])
def get_imgs(img_path, bbox=None, transform=image_transform, normalize=norm):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    ww = int(256 / 4)

    LR = F.interpolate(img, size=(ww, ww), mode='bilinear')
    return LR,img


img_path="/opt/data/private/workspace/GALIP-main/code/src/samples/Eastern_Towhee_0112_22231.jpg"
LR,HR=get_imgs(img_path, bbox=None, transform=image_transform, normalize=norm)

names="LR"
vutils.save_image(LR.data, '../samples/%s.png'%(names), nrow=8, value_range=(-1, 1), normalize=True)
vutils.save_image(HR.data, '../samples/%s.png'%("HR"), nrow=8, value_range=(-1, 1), normalize=True)




captions = ['it is a dog.']

mkdir_p('./samples')

# generate from text
with torch.no_grad():
    for i in range(len(captions)):
        caption = captions[i]
        tokenized_text = clip.tokenize([caption]).to(device)
      
        sent_emb, word_emb = text_encoder(tokenized_text)
        
        sent_emb = sent_emb.repeat(batch_size,1)
     
        fake_imgs = netG(LR,sent_emb,eval=True).float()
        name = f'{captions[i].replace(" ", "-")}'
        if not os.path.exists("../samples"):
            os.makedirs("../samples")
        
        vutils.save_image(fake_imgs.data, '../samples/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)