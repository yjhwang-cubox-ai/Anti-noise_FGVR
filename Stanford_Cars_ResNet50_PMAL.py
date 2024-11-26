from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.models
from sam import SAM
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import imgaug as ia
import imgaug.augmenters as iaa
from vic.loss import CharbonnierLoss
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from basic_conv import *
from example.model.smooth_cross_entropy import smooth_crossentropy
from example.utility.bypass_bn import enable_running_stats, disable_running_stats
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def test(net, criterion, batch_size, test_path):
    net.eval()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=4,
                                           pin_memory=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            inputs, targets = inputs.to(device), targets.to(device)
            output_1, output_2, output_3, output_ORI, map1, map2, map3 = net(inputs)

            outputs_com = output_1 + output_2 + output_3 + output_ORI

            loss = criterion(output_ORI, targets).mean()

            test_loss += loss.item()
            _, predicted = torch.max(output_ORI.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1),
                100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss




class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3


class Anti_Noise_Decoder(nn.Module):
    def __init__(self, scale, in_channel):
        super(Anti_Noise_Decoder, self).__init__()
        self.Sigmoid = nn.Sigmoid()

        in_channel = in_channel // (scale * scale)

        self.skip = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)

        )

        self.process = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channel, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, map):
        return self.skip(x) + self.process(map)

class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class, classifier):
        super().__init__()
        self.Features = Features(net_layers)
        self.classifier_pool = nn.Sequential(classifier[0])
        self.classifier_initial = nn.Sequential(classifier[1])
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=56, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=14, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )


    def forward(self, x):
        x1, x2, x3 = self.Features(x)
        map1 = x1.clone()
        map2 = x2.clone()
        map3 = x3.clone()

        classifiers = self.classifier_pool(x3).view(x3.size(0), -1)
        classifiers = self.classifier_initial(classifiers)

        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)


        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        return x1_c, x2_c, x3_c, classifiers, map1, map2, map3

def img_add_noise(x, transformation_seq):

    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy()
    x = transformation_seq(images=x)
    x = torch.from_numpy(x.astype(np.float32))
    x = x.permute(0, 3, 1, 2)
    return x


def CELoss(x, y):
    return smooth_crossentropy(x, y, smoothing=0.1)


def train(nb_epoch, batch_size, store_name, num_class=0, start_epoch=0, data_path=''):

    alpha = 1

    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)


    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=data_path+'/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # 시스템에 맞게 조정
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


    # CUDA 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    # Mixed Precision Training을 위한 scaler 초기화
    scaler = GradScaler()

    # Tensorboard 설정
    writer = SummaryWriter(f'runs/{store_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # 모델 초기화
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features, num_class)

    net_layers = list(net.children())
    classifier = net_layers[8:10]
    net_layers = net_layers[0:8]
    net = Network_Wrapper(net_layers, num_class, classifier)

    # Multi-GPU 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netp = torch.nn.DataParallel(net)
    else:
        netp = net

    netp = netp.to(device)
    decoder1 = Anti_Noise_Decoder(1, 512).to(device)
    decoder2 = Anti_Noise_Decoder(2, 1024).to(device)
    decoder3 = Anti_Noise_Decoder(4, 2048).to(device)

    CB_loss = CharbonnierLoss()

    base_optimizer = torch.optim.SGD


    optimizer = SAM([
        {'params': net.classifier_initial.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},

        {'params': decoder1.skip.parameters(), 'lr': 0.002},
        {'params': decoder1.process.parameters(), 'lr': 0.002},
        {'params': decoder2.skip.parameters(), 'lr': 0.002},
        {'params': decoder2.process.parameters(), 'lr': 0.002},
        {'params': decoder3.skip.parameters(), 'lr': 0.002},
        {'params': decoder3.process.parameters(), 'lr': 0.002},

        {'params': net.Features.parameters(), 'lr': 0.0002}

    ],
        base_optimizer, adaptive=False, momentum=0.9, weight_decay=5e-4)



    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002,  0.002, 0.002,  0.002, 0.002,  0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        print(f'\nEpoch: {epoch}/{nb_epoch}')
        net.train()
        
        # Progress bar 초기화
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{nb_epoch}')
        
        train_loss = 0
        train_loss1 = train_loss2 = train_loss3 = train_loss4 = train_loss5 = 0
        correct = total = 0
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if inputs.shape[0] < batch_size:
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Learning rate 조정
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            sometimes_1 = lambda aug: iaa.Sometimes(0.2, aug)
            sometimes_2 = lambda aug: iaa.Sometimes(0.5, aug)



            trans_seq_aug = iaa.Sequential(
                [

                    sometimes_1(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-15, 15),
                        shear=(-15, 15),
                        order=[0, 1],
                        cval=(0, 1),
                        mode=ia.ALL
                    )),
                    sometimes_2(iaa.GaussianBlur((0, 3.0)))
                ],
                random_order=True
            )

            trans_seq = iaa.Sequential(
                [

                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    )
                ],
                random_order=True
            )

            # Mixed Precision Training with H1-H4 steps
            # H1
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                enable_running_stats(netp)
                optimizer.zero_grad()
                inputs1_gt = img_add_noise(inputs, trans_seq_aug).to(device)
                inputs1 = img_add_noise(inputs1_gt, trans_seq).to(device)
                output_1, _, _, _, map1, _, _ = netp(inputs1)
                loss1_c = CELoss(output_1, targets).mean()
                
                inputs1_syn = decoder1(inputs1, map1)
                loss1_g = CB_loss(inputs1_syn, inputs1_gt)
                
                output_1_syn, _, _, _, _, _, _ = netp(inputs1_syn)
                loss1_c_syn = CELoss(output_1_syn, targets).mean()
                
                loss1 = loss1_c + (alpha * loss1_g) + loss1_c_syn

            scaler.scale(loss1).backward(retain_graph=True)
            optimizer.first_step(zero_grad=True)

            # H1 second step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                disable_running_stats(netp)
                output_1, _, _, _, map1, _, _ = netp(inputs1)
                loss1_c = CELoss(output_1, targets).mean()
                
                inputs1_syn = decoder1(inputs1, map1)
                loss1_g = CB_loss(inputs1_syn, inputs1_gt)
                
                output_1_syn, _, _, _, _, _, _ = netp(inputs1_syn)
                loss1_c_syn = CELoss(output_1_syn, targets).mean()
                
                loss1_ = loss1_c + (alpha * loss1_g) + loss1_c_syn

            scaler.scale(loss1_).backward()
            optimizer.second_step(zero_grad=True)

            # H2
            # H2 first forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                enable_running_stats(netp)
                optimizer.zero_grad()
                inputs2_gt = img_add_noise(inputs, trans_seq_aug).to(device)
                inputs2 = img_add_noise(inputs2_gt, trans_seq).to(device)
                _, output_2, _, _, _, map2, _ = netp(inputs2)
                loss2_c = CELoss(output_2, targets).mean()
                
                inputs2_syn = decoder2(inputs2, map2)
                loss2_g = CB_loss(inputs2_syn, inputs2_gt)
                
                _, output_2_syn, _, _, _, _, _ = netp(inputs2_syn)
                loss2_c_syn = CELoss(output_2_syn, targets).mean()
                
                loss2 = loss2_c + (alpha * loss2_g) + loss2_c_syn

            scaler.scale(loss2).backward(retain_graph=True)
            optimizer.first_step(zero_grad=True)

            # H2 second forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                disable_running_stats(netp)
                output_2, _, _, _, _, map2, _ = netp(inputs2)
                loss2_c = CELoss(output_2, targets).mean()
                
                inputs2_syn = decoder2(inputs2, map2)
                loss2_g = CB_loss(inputs2_syn, inputs2_gt)
                
                _, output_2_syn, _, _, _, _, _ = netp(inputs2_syn)
                loss2_c_syn = CELoss(output_2_syn, targets).mean()
                
                loss2_ = loss2_c + (alpha * loss2_g) + loss2_c_syn

            scaler.scale(loss2_).backward()
            optimizer.second_step(zero_grad=True)

            #H3
            # H3 first forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                enable_running_stats(netp)
                optimizer.zero_grad()
                inputs3_gt = img_add_noise(inputs, trans_seq_aug).to(device)
                inputs3 = img_add_noise(inputs3_gt, trans_seq).to(device)
                _, _, output_3, _, _, _, map3 = netp(inputs3)
                loss3_c = CELoss(output_3, targets).mean()
                
                inputs3_syn = decoder3(inputs3, map3)
                loss3_g = CB_loss(inputs3_syn, inputs3_gt)
                
                _, _, output_3_syn, _, _, _, _ = netp(inputs3_syn)
                loss3_c_syn = CELoss(output_3_syn, targets).mean()
                
                loss3 = loss3_c + (alpha * loss3_g) + loss3_c_syn

            scaler.scale(loss3).backward(retain_graph=True)
            optimizer.first_step(zero_grad=True)

            # H3 second forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                disable_running_stats(netp)
                _, _, output_3, _, _, _, map3 = netp(inputs3)
                loss3_c = CELoss(output_3, targets).mean()
                
                inputs3_syn = decoder3(inputs3, map3)
                loss3_g = CB_loss(inputs3_syn, inputs3_gt)
                
                _, _, output_3_syn, _, _, _, _ = netp(inputs3_syn)
                loss3_c_syn = CELoss(output_3_syn, targets).mean()
                
                loss3_ = loss3_c + (alpha * loss3_g) + loss3_c_syn

            scaler.scale(loss3_).backward()
            optimizer.second_step(zero_grad=True)



            # H4
            # H4 first forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                enable_running_stats(netp)
                optimizer.zero_grad()
                output_1_final, output_2_final, output_3_final, output_ORI, _, _, _ = netp(inputs)
                ORI_loss = CELoss(output_1_final, targets).mean() + \
                                    CELoss(output_2_final, targets).mean() + \
                                    CELoss(output_3_final, targets).mean() + \
                                    CELoss(output_ORI, targets).mean() * 2

            scaler.scale(ORI_loss).backward(retain_graph=True)
            optimizer.first_step(zero_grad=True)

            # H4 second forward-backward step
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                disable_running_stats(netp)
                output_1_final, output_2_final, output_3_final, output_ORI, _, _, _ = netp(inputs)
                ORI_loss_ = CELoss(output_1_final, targets).mean() + \
                              CELoss(output_2_final, targets).mean() + \
                              CELoss(output_3_final, targets).mean() + \
                              CELoss(output_ORI, targets).mean() * 2

            scaler.scale(ORI_loss_).backward()
            optimizer.second_step(zero_grad=True)


            # Update metrics
            _, predicted = torch.max(output_ORI.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # Loss accumulation
            train_loss += (loss1.item() + loss2.item() + loss3.item() + ORI_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += (loss1_g.item() + loss2_g.item() + loss3_g.item())
            train_loss5 += ORI_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*float(correct)/total:.2f}%'
            })

            # Log to tensorboard
            writer.add_scalar('Batch/Loss', train_loss/(batch_idx+1), epoch * len(trainloader) + batch_idx)
            writer.add_scalar('Batch/Accuracy', 100.*float(correct)/total, epoch * len(trainloader) + batch_idx)

        # Epoch end operations
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / len(trainloader)
        
        # Validation
        val_acc_com, val_loss = test(net, CELoss, batch_size, data_path+'/test')
        
        # Log to tensorboard
        writer.add_scalars('Epoch', {
            'Train_Loss': train_loss,
            'Train_Acc': train_acc,
            'Val_Loss': val_loss,
            'Val_Acc': val_acc_com
        }, epoch)

        # Save best model
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'decoder1': decoder1.state_dict(),
                'decoder2': decoder2.state_dict(),
                'decoder3': decoder3.state_dict(),
                'optimizer': optimizer.state_dict(),
                'max_val_acc': max_val_acc,
            }, store_name)

    writer.close()

def save_checkpoint(state, store_name):
    """Save checkpoint"""
    filepath = os.path.join(store_name, 'checkpoint.pth')
    torch.save(state, filepath)

if __name__ == '__main__':
    data_path = '/data/Stanford_Cars'
    if not os.path.isdir('results'):
        os.mkdir('results')
    train(nb_epoch=200,             # number of epoch
             batch_size=8,         # batch size
             store_name='results/Stanford_Cars_ResNet50_PMAL',     # the folder for saving results
             num_class=196,          # number of categories
             start_epoch=0,         # the start epoch number
             data_path = data_path)   # the path to the dataset

