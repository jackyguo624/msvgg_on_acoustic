import torch.nn as nn
import math
from torch import FloatTensor
from torch.autograd import Variable

__all__ = [
    'vgg22', 'vgg23', 'vgg24', 'vgg32', 'vgg33', 'vgg34', 'vgg42',
    'vgg43', 'vgg44', 'vgg55', 'vgg66', 'vgg77'
]


class MSVGG(nn.Module):
    def __init__(self, features, num_class, special_feature_map=False):
        super(MSVGG, self).__init__()
        self.features = features
        if special_feature_map:
            first_linear = nn.Linear(512*2*5, 2048)
        else:
            first_linear = nn.Linear(512*2*3, 2048)
        self.classifier = nn.Sequential(
            first_linear,
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, num_class)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, out_cfg, kernel_size, batch_norm=False):
    layers = []
    in_channels = 3
    for v, output in zip(cfg, out_cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=(v[1], v[2]), stride=(v[1], v[2]))]
        else:
            conv2d = nn.Conv2d(in_channels, output, kernel_size=kernel_size, padding=(v[1], v[2]))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(output)]
            else:
                layers += [conv2d]
            in_channels = output
    return nn.Sequential(*layers)

out_cfg = [64, 64, -1,
           128, 128, -1,
           256, 256, 256, -1,
           512, 512, 512, -1]

cfg = {
    'vgg22': [('C', 1, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 2),
              ('C', 1, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg23': [('C', 1, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg24': [('C', 1, 1), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 1), ('C', 1, 1), ('M', 2, 1),
              ('C', 1, 1), ('C', 1, 1), ('C', 0, 0), ('M', 2, 1),  # ('C', 1, 1), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)
              ('C', 1, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg32': [('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('C', 0, 0), ('M', 2, 2),
              ('C', 1, 0), ('C', 1, 0), ('C', 1, 0), ('M', 2, 1)],

    'vgg33': [('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('C', 1, 0), ('M', 2, 1)],

    'vgg34': [('C', 1, 1), ('C', 1, 1), ('M', 2, 1),
              ('C', 1, 1), ('C', 1, 1), ('M', 2, 1),
              ('C', 1, 1), ('C', 1, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 1, 0), ('C', 1, 0), ('C', 1, 0), ('M', 2, 1)],

    'vgg42': [('C', 3, 0), ('C', 0, 1), ('M', 2, 1), # ('C', 3, 0), ('C', 0, 2), ('M', 2, 1),
              ('C', 3, 0), ('C', 3, 0), ('M', 2, 1), # ('C', 3, 0), ('C', 2, 0), ('M', 2, 1),
              ('C', 3, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 2),
              ('C', 3, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg43': [('C', 3, 0), ('C', 2, 0), ('M', 2, 1),
              ('C', 3, 0), ('C', 2, 0), ('M', 2, 1),
              ('C', 3, 0), ('C', 2, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 3, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg44': [('C', 3, 1), ('C', 2, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 2, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 2, 0), ('C', 0, 0), ('M', 2, 1),
              ('C', 3, 0), ('C', 0, 0), ('C', 0, 0), ('M', 2, 1)],

    'vgg55': [('C', 3, 1), ('C', 3, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 2, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 1, 1), ('C', 0, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 3, 1), ('C', 0, 1), ('M', 2, 1)],

    'vgg66': [('C', 4, 2), ('C', 3, 2), ('M', 2, 1),
              ('C', 3, 2), ('C', 2, 2), ('M', 2, 1),
              ('C', 3, 2), ('C', 2, 1), ('C', 2, 1), ('M', 2, 1),
              ('C', 3, 1), ('C', 3, 1), ('C', 1, 1), ('M', 2, 1)],

    'vgg77': [('C', 4, 2), ('C', 4, 2), ('M', 2, 1),
              ('C', 3, 2), ('C', 2, 2), ('M', 2, 1),
              ('C', 3, 2), ('C', 3, 2), ('C', 2, 2), ('M', 2, 1),
              ('C', 3, 2), ('C', 3, 2), ('C', 3, 2), ('M', 2, 1)]
}


def vgg22(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg22'], out_cfg, (2, 2)), num_class=1920, special_feature_map=True)
    return model


def vgg23(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg23'], out_cfg, (2, 3)), num_class=1920)
    return model


def vgg24(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg24'], out_cfg, (2, 4)), num_class=1920)
    return model


def vgg32(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg32'], out_cfg, (3, 2)), num_class=1920, special_feature_map=True)
    return model


def vgg33(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg33'], out_cfg, (3, 3)), num_class=1920)
    return model

def vgg34(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg34'], out_cfg, (3, 4)), num_class=1920)
    return model

def vgg42(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg42'], out_cfg, (4, 2)), num_class=1920)
    return model

def vgg43(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg43'], out_cfg, (4, 3)), num_class=1920)
    return model

def vgg44(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg44'], out_cfg, (4, 4)), num_class=1920)
    return model

def vgg55(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg55'], out_cfg, (5, 5)), num_class=1920)
    return model

def vgg66(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg66'], out_cfg, (6, 6)), num_class=1920)
    return model

def vgg77(**kwargs):
    model = MSVGG(features=make_layers(cfg['vgg77'], out_cfg, (7, 7)), num_class=1920)
    return model

if __name__ == '__main__':
    a = Variable(FloatTensor(2,3,40,23), requires_grad=True)
    print FloatTensor(2,3,40,23).size()
    myvgg22 = vgg22()
    myvgg22.forward(a)

    myvgg23 = vgg23()
    myvgg23.forward(a)

    myvgg24 = vgg24()
    myvgg24.forward(a)

    myvgg32 = vgg32()
    myvgg32.forward(a)
    
    myvgg33 = vgg33()
    myvgg33.forward(a)

    myvgg34 = vgg34()
    myvgg34.forward(a)

    myvgg42 = vgg42()
    myvgg42.forward(a)

    myvgg43 = vgg43()
    myvgg43.forward(a)

    myvgg44 = vgg44()
    myvgg44.forward(a)

    myvgg55 = vgg55()
    myvgg55.forward(a)

    myvgg66 = vgg66()
    myvgg66.forward(a)

    myvgg77 = vgg77()
    myvgg77.forward(a)


