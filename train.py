from reIDdataset import ReIDdataset
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from network import StructureEncoder, Decoder, Discriminator, AppearanceEncoder
import argparse
import torch.backends.cudnn as cudnn
import torch
from torchvision.utils import save_image
import numpy.random as random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_root',default='C:/Users/wooks/Desktop/code/', type=str,help='data directory')

opts = parser.parse_args()
data_root = opts.data_root
str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gpu_ids.append(int(str_id))
num_gpu = len(gpu_ids)

cudnn.benchmark = True
output_dir = opts.output_path

random.seed(7) #fix random result

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_list = [transforms.ToTensor(),
                    transforms.Normalize(mean,
                                        std)]
transform_list = [transforms.RandomCrop((256, 128))]
transform_list = [transforms.Pad(10, padding_mode='edge')]
transform_list = [transforms.Resize((256,128), interpolation=3)]
transform_list = [transforms.RandomHorizontalFlip()]
transform = transforms.Compose(transform_list)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    return tensor * std + mean

dataset = ReIDdataset(data_root, transform = transform)
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)

train_loader_a, train_loader_b, test_loader_a, test_loader_b = dataloader

train_a_rand = random.permutation(train_loader_a.dataset.img_num)
train_b_rand = random.permutation(train_loader_b.dataset.img_num)
test_a_rand = random.permutation(test_loader_a.dataset.img_num)
test_b_rand = random.permutation(test_loader_b.dataset.img_num)

train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in train_a_rand]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in train_b_rand]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in test_a_rand]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in test_b_rand]).cuda()

output_directory = os.path.join(opts.output_path + "/outputs", opts.name)
image_directory = os.path.join(output_directory, 'images')
checkpoint_directory = os.path.join(output_directory, 'checkpoint')

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)


nepoch = 0

st_enc_a = StructureEncoder()
st_enc_b = st_enc_a

ap_enc_a = AppearanceEncoder()
ap_enc_b = ap_enc_a

gen_a = Decoder()
gen_b = gen_a

dis_a = Discriminator()
dis_b = dis_a

dis_params = dis_a.parameters()
gen_params = gen_a.parameters()

dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=0.0001)
gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=0.0001)


def recon_criterion(original, gen_img):
    return torch.mean(torch.abs(original - gen_img))

id_criterion = nn.CrossEntropyLoss()


while True:
    for it, ((images_a,labels_a, pos_a),  (images_b, labels_b, pos_b)) in enumerate(zip(train_loader_a, train_loader_b)):

        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        pos_a, pos_b = pos_a.cuda().detach(), pos_b.cuda().detach()
        labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()
        dis_opt.zero_grad()

        # Main training code
        st_a = st_enc_a.forward(images_a)
        st_b = st_enc_b.forward(images_b)

        ap_b = ap_enc_b.forward(images_b) #s_id
        ap_a = ap_enc_a.forward(images_a)

        a2b = gen_a.forward(st_a, ap_b) #adv loss, c_id, code1_recon, code2_recon
        b2a = gen_b.forward(st_b, ap_a)
        
        recon_b = gen_b.forward(st_b, ap_b) #img1 recon loss
        recon_a = gen_a.forward(st_a, ap_a)

        ap_pb = ap_enc_b.forward(pos_b) #s_id
        ap_pa = ap_enc_a.forward(pos_a)

        recon_bp = gen_b.forward(st_b, ap_pb) #img2 recon loss
        recon_ap = gen_a.forward(st_a, ap_pa)

        recon_ap_a = ap_enc_a.forward(a2b)
        recon_ap_b = ap_enc_b.forward(b2a)

        recon_st_a = st_enc_a.forward(a2b)
        recon_st_b = st_enc_a.forward(b2a)


        loss_gen_a1 = recon_criterion(images_a, a2b)
        loss_gen_a2 = recon_criterion(images_a, recon_ap)

        loss_gen_b1 = recon_criterion(images_b, b2a)
        loss_gen_b2 = recon_criterion(images_b, recon_bp)
        
        # Code reconstruction loss
        loss_ap_a = recon_criterion(ap_a, recon_ap_a)
        loss_ap_b = recon_criterion(ap_b, recon_ap_b)
        
        loss_st_a = recon_criterion(st_a, recon_st_a)
        loss_st_b = recon_criterion(st_b, recon_st_b)

        id_a = id_criterion(a2b,labels_a)
        id_b = id_criterion(b2a,labels_b)

        loss_gen_adv_a = dis_a.calc_gen_loss(dis_a, b2a)
        loss_gen_adv_b = dis_b.calc_gen_loss(dis_b, a2b)

        recon_loss_total = loss_gen_a1 + loss_gen_a2 + loss_gen_b1 + loss_gen_b2 + loss_ap_a + loss_ap_b + loss_st_a + loss_st_b + \
                           loss_gen_adv_a + loss_gen_adv_b + id_a + id_b

        loss_dis_a, reg_a = dis_a.calc_dis_loss(dis_a, b2a.detach(), images_a)
        loss_dis_b, reg_b = dis_b.calc_dis_loss(dis_b, a2b.detach(), images_b)
        
        dis_loss_total = loss_dis_a + loss_dis_b

        
        print(f"recon_loss : {recon_loss_total} \t dis_loss : {dis_loss_total}")

        recon_loss_total.backward()
        dis_loss_total.backward()

        gen_opt.step()
        dis_opt.step()

        if (it + 1) % 100 == 0:
            with torch.no_grad():
                save_img = a2b.detach()
                save_img = denormalize(save_img)
                
            save_image(save_img, os.path.join(output_dir, f"save_img_{nepoch}.png"))

        it += 1

    nepoch = nepoch+1
    if(nepoch + 1) % 10 == 0:
        torch.save({'a': gen_a.state_dict()}, output_directory+f"gen_{nepoch}")
        torch.save({'a': dis_a.state_dict()}, output_directory+f"gen_{nepoch}")
        torch.save({'gen': gen_opt.state_dict(), 'dis': dis_opt.state_dict()}, output_directory+f"gen_{nepoch}")

