import torch
import torch.nn as nn
import torchvision.models as models

###### Encoder #########
class StructureEncoder(nn.Module):
    def __init__(self):
        super(StructureEncoder, self).__init__()
        
        # Conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv1: 1 x 256 x 128 -> 16 x 128 x 64
            nn.InstanceNorm2d(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv2: 16 x 128 x 64 -> 32 x 128 x 64
            nn.InstanceNorm2d(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # Conv3: 32 x 128 x 64 -> 32 x 128 x 64
            nn.InstanceNorm2d(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv4: 32 x 64 x 32 -> 64 x 64 x 32
            nn.InstanceNorm2d(),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[self._make_res_block(64) for _ in range(4)]  # ResBlocks: 64 x 64 x 32 -> 64 x 64 x 32
        )
        
        # ASPP module
        self.aspp = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # ASPP Conv1: 64 x 64 x 32 -> 32 x 64 x 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # ASPP Conv2: 32 x 64 x 32 -> 32 x 64 x 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),  # ASPP Conv3: 32 x 64 x 32 -> 32 x 64 x 32
            nn.ReLU()
        )
        
        # Final Conv layer
        self.final_conv = nn.Conv2d(32, 128, kernel_size=1)  # Conv5: 32 x 64 x 32 -> 128 x 64 x 32
        
    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.res_blocks(x)
        x = self.aspp(x)
        x = self.final_conv(x)
        return x

def AdaIn(content, style, eps=1e-5):
    # content: 콘텐츠 이미지의 특성 맵
    # style: 스타일 이미지의 특성 맵
    
    # 콘텐츠 이미지의 평균과 표준편차 계산
    c_mean = content.mean(dim=(2, 3), keepdim=True)
    c_std = content.std(dim=(2, 3), keepdim=True) + eps
    
    # 스타일 이미지의 평균과 표준편차 계산
    s_mean = style.mean(dim=(2, 3), keepdim=True)
    s_std = style.std(dim=(2, 3), keepdim=True) + eps
    
    # Instance Normalization 후, 스타일의 통계적 특징을 적용
    normalized_content = (content - c_mean) / c_std
    return normalized_content * s_std + s_mean

class AdaINLayer(nn.Module):
    def __init__(self):
        super(AdaINLayer, self).__init__()
    
    def forward(self, content, style):
        return AdaIn(content, style)

# Appearance Encoder (ResNet50 기반)
class AppearanceEncoder(nn.Module):
    def __init__(self):
        super(AppearanceEncoder, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        # ResNet50의 마지막 두 레이어(Global Pooling, FC Layer) 제거
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.adain = AdaINLayer()
        
    def forward(self, x, style):

        content_features = self.backbone(x)

        adain_features = self.adain(content_features, style)

        pooled_features = self.pool(adain_features)
        
        return pooled_features.squeeze()
###### Encoder #########



###### Decoder #########
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.res_blocks = nn.Sequential(
            *[self._make_res_block(128) for _ in range(4)]  # ResBlock: 128 x 64 x 32
        )
        
        # Upsample + Conv1
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample: 128 x 64 x 32 -> 128 x 128 x 64
        self.conv1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)  # Conv1: 128 x 128 x 64 -> 64 x 128 x 64
  
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample: 64 x 128 x 64 -> 64 x 256 x 128
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  # Conv2: 64 x 256 x 128 -> 32 x 256 x 128

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Conv3: 32 x 256 x 128 -> 32 x 256 x 128
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Conv4: 32 x 256 x 128 -> 32 x 256 x 128
        self.conv5 = nn.Conv2d(32, 3, kernel_size=1)  # Conv5: 32 x 256 x 128 -> 3 x 256 x 128 (최종 출력)
        
    def _make_res_block(self, channels):
        """Residual Block 정의"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):

        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)
        
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Conv1: 3 x 256 x 128 -> 32 x 256 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        
        # Conv2: 32 x 256 x 128 -> 32 x 256 x 128
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # Conv3: 32 x 256 x 128 -> 32 x 128 x 64
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        # Conv4: 32 x 128 x 64 -> 32 x 128 x 64
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        # Conv5: 32 x 128 x 64 -> 64 x 64 x 32
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu5 = nn.ReLU()
        
        # Residual Blocks (64 x 64 x 32 -> 64 x 64 x 32)
        self.res_blocks = nn.Sequential(
            *[self._make_res_block(64) for _ in range(4)]  # 4개의 Residual Block
        )
        
        # Conv6: 64 x 64 x 32 -> 1 x 64 x 32
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        
    def _make_res_block(self, channels):
        """Residual Block 정의"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def calc_dis_loss(self, model, real, fake):
        real.requires_grad_()
        dis_fake = model.forward(fake)
        dis_real = model.forward(real)

        loss = 0

        for _, (dis_fake_, dis_real_) in enumerate(zip(dis_fake, dis_real)):
            loss += torch.mean((dis_fake_)**2) + torch.mean((dis_real)**2)
            reg += self.compute_grad2(dis_real, real).mean()

        loss = loss + reg
        return reg
    
    def calc_gen_loss(self, model, fake):
        dis_fake = model.forward(fake)
        loss = 0

        for _, (dis_fake_) in enumerate(dis_fake):
            loss += torch.mean((dis_fake_ - 1)**2) * 2  # LSGAN

        return loss

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.res_blocks(x)
        x = self.conv6(x)
        
        return x