import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn.functional as F
from heatmap import *

sys.path.insert(0, '../')
from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets import get_fix_data
from PIL import Image
import numpy as np
from lib.SemanyicMatch import SemanticMatchingModel

from lib.utils import load_model_weights,mkdir_p
from models.GALIP import NetG, CLIP_TXT_ENCODER,CLIP_IMG_ENCODER
from lib.perpare import load_clip
device = 'cpu' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()



text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)

image_encoder=CLIP_IMG_ENCODER(clip_model).to(device)
# CLIP4trn = load_clip("ViT-B/32", device).eval()
# CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
# for p in CLIP_img_enc.parameters():
#     p.requires_grad = False
# CLIP_img_enc.eval()

netG = NetG(64, 512, 512, 256, 3, False, clip_model).to(device)
path = '../saved_models/pretrained/pre_cc12m.pth'
path = '/opt/data/private/carr/code/saved_models/cele/GALIP_nf64_normal_cele_256_2024_12_12_01_48_30/state_epoch_095.pth'
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


img_path="/opt/data/private/carr/code/src/samples/Eastern_Towhee_0112_22231.jpg"
LR,HR=get_imgs(img_path, bbox=None, transform=image_transform, normalize=norm)

names="LR"
vutils.save_image(LR.data, './samples/%s.png'%(names), nrow=8, value_range=(-1, 1), normalize=True)
vutils.save_image(HR.data, './samples/%s.png'%("HR"), nrow=8, value_range=(-1, 1), normalize=True)




captions = ['the-bird-has-big-beak-when-compared-to-its-body,-it-has-red-crown-nape,-and-red-tarsus-and-feet.']
captions = ['*']
mkdir_p('./samples')

# generate from text
with torch.no_grad():
    for i in range(len(captions)):
        caption = captions[i]
        tokenized_text = clip.tokenize([caption]).to(device)
      
        sent_emb, word_emb = text_encoder(tokenized_text)
        print(type(sent_emb),type(word_emb))
        
        CLIP_real,real_emb=image_encoder(LR)
        print(type(CLIP_real))
      
        similarity = F.cosine_similarity(real_emb, sent_emb, dim=1)
        print(similarity)
        sent_emb = sent_emb.repeat(batch_size,1)

        fake_imgs = netG(LR,sent_emb,eval=True).float()
        name = f'{captions[i].replace(" ", "-")}'
        if not os.path.exists("../samples"):
            os.makedirs("../samples")
        
        vutils.save_image(fake_imgs.data, '../samples/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)

target_layers = [netG.GBlocks[-1]]
# img, data = image_proprecess(imgs_path)
img=LR

cam = GradCAM(model=netG, target_layers=target_layers)
target_category = None

# data = data.cuda()
grayscale_cam = cam(input_tensor=LR,c=sent_emb, target_category=target_category)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(np.array(img) / 255.,
                                    grayscale_cam,
                                    use_rgb=True)
print("_______________finish_______________")
plt.imshow(visualization)
plt.xticks()
plt.yticks()
plt.axis('off')
plt.savefig("/opt/data/private/carr/code/src/samples/01.jpg")
print("_______________finish_______________")

