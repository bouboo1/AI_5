import torch.nn as nn
from torchvision.models import resnet50

# 使用resnet 50
class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )
        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],
            nn.Flatten()
        )
        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)
        feature = self.resnet_p(hidden_state)
        return self.hidden_trans(hidden_state), self.trans(feature)