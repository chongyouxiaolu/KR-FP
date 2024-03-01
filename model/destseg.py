import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer


class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

    def forward(self, x, fp):
        x1, x2, x3, x4 = self.encoder(x)
        if not self.ed:
            return (x1, x2, x3)
        x = x4
        if fp == 0.3:
            b4 = self.decoder_layer4(nn.Dropout2d(0.3)(x))
            b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
            b3 = self.decoder_layer3(nn.Dropout2d(0.3)(b3))
            b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
            b2 = self.decoder_layer2(nn.Dropout2d(0.3)(b2))
            b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
            b1 = self.decoder_layer1(nn.Dropout2d(0.3)(b1))
            return (b1, b2, b3)
        if fp == 0.4:
            b4 = self.decoder_layer4(nn.Dropout2d(0.4)(x))
            b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
            b3 = self.decoder_layer3(nn.Dropout2d(0.4)(b3))
            b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
            b2 = self.decoder_layer2(nn.Dropout2d(0.4)(b2))
            b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
            b1 = self.decoder_layer1(nn.Dropout2d(0.4)(b1))
            return (b1, b2, b3)
        if fp == 0.5:
            b4 = self.decoder_layer4(nn.Dropout2d(0.5)(x))
            b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
            b3 = self.decoder_layer3(nn.Dropout2d(0.5)(b3))
            b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
            b2 = self.decoder_layer2(nn.Dropout2d(0.5)(b2))
            b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
            b1 = self.decoder_layer1(nn.Dropout2d(0.5)(b1))
            return (b1, b2, b3)
        if fp == 0.7:
            b4 = self.decoder_layer4(nn.Dropout2d(0.7)(x))
            b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
            b3 = self.decoder_layer3(nn.Dropout2d(0.7)(b3))
            b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
            b2 = self.decoder_layer2(nn.Dropout2d(0.7)(b2))
            b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
            b1 = self.decoder_layer1(nn.Dropout2d(0.7)(b1))
            return (b1, b2, b3)
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class DeSTSeg(nn.Module):
    def __init__(self, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)
        self.shapes = [16, 32, 64]
        abfs = nn.ModuleList()
        in_channels = [64, 128, 256]
        out_channels = [64, 128, 256]
        mid_channel = 256

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))

        self.abfs = abfs[::-1]


    def forward(self, img_aug, img_origin=None, fp=0):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        fteacher = self.teacher_net(img_origin)
        student_features = self.student_net(img_aug,fp)
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)
        loss_reviewkd = hcl(results, fteacher) * 1.0

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug,fp=0)
        ]
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )

        output_segmentation = self.segmentation_net(output)

        if self.dest:
            outputs_student = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug,fp)
        ]
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin,fp)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]

        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)

        return output_segmentation, output_de_st, output_de_st_list, loss_reviewkd

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all