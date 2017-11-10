from dataloader import Dataloader
from msvgg import vgg22,vgg33,vgg44
import torch
import argparse
from torch.autograd import Variable
from gpuselect import gpu_select
import modulesave
import os.path

batch_size=128


mdl_path='./best_module22.pt'
device_ids = gpu_select(2)
first_device=device_ids[0]
net = vgg22()
net = torch.nn.DataParallel(net,device_ids=device_ids)
net.cuda(first_device)

base = '/home/slhome/jqg01/work-home/workspace/pytorch/msvgg_on_acoustic'
f_train = base+'/mfcc40_23/train/feats/expand_feats.{0:d}.ark'
ali_train = base+'/ali/ali.ark'
f_dev = base+'/mfcc40_23/dev/feats/expand_feats.{0:d}.ark'
ali_dev = base + '/dev_ali/ali.ark'


change_lr=False
lr=1e-4
pre_acc = 0.0
best_acc = 0.0
lr_adaptive=True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)

running_loss = 0.0
for epoch in range(50):

    dl = Dataloader(feature=f_train,
                ali=ali_train,ext_data_n=20,batch_size=batch_size)

    devloader = Dataloader(feature=f_dev,
                           ali=ali_dev,ext_data_n=20,batch_size=batch_size)
    
    '''check if need to change learning rate'''
    for param_group in optimizer.param_groups:
        if change_lr:
            lr = lr * 0.5
            param_group['lr'] = lr
            change_lr = False
        print("Epoch: {0:d} Optimizer's lr: {1:f}\n"
              .format(epoch, param_group['lr']))
    if lr <= 1e-10:
        print 'lr <= 1e-10 stop training!'
        break

    correct = 0.0
    total = 0.0
    running_loss = 0.0

    for i, data in enumerate(dl, 0):
        inputs, labels = data
        bs, datadim = inputs.size()
        inputs = inputs.view(bs, 3*23, -1)
        i_tensor = torch.cat((inputs[:,::3],inputs[:,1::3],inputs[:,2::3])
                             ,dim=1).view(bs,3,23,-1)

        
        l_tensor = labels
        (f, l)=(Variable(i_tensor.cuda(first_device)), 
               Variable(l_tensor.cuda(first_device)))
        optimizer.zero_grad()
        outputs = net(f)
        loss = criterion(outputs, l)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
    running_loss = 0.0
    net.eval()
    for j, data in enumerate(devloader,0):
        inputs, labels = data
        bs, datadim = inputs.size()
        inputs = inputs.view(bs, 3*23, -1)
        i_tensor = torch.cat((inputs[:,::3],inputs[:,1::3],inputs[:,2::3])
                             ,dim=1).view(bs,3,23,-1)


        l_tensor=labels
        (inputs_var,labels_var)=(Variable(i_tensor.cuda(first_device)),
               Variable(l_tensor.cuda(first_device)))
        
        outputs_var = net(inputs_var)
        loss = criterion(outputs_var, labels_var)
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs_var.data, 1)
        total += labels.size(0)
        correct += (predicted == l_tensor.cuda(first_device)).sum()

    cur_acc = 100 * correct / total
    if lr_adaptive and cur_acc < pre_acc:
        change_lr = True
    pre_acc = cur_acc
    print("Epoch : {0:d} , cv_loss : {1:f}, cv_acc: {2:f} % \n"
          .format(epoch, running_loss/j, cur_acc))

    if cur_acc > best_acc:
        modulesave.save_module(net, epoch, cur_acc, device_ids, mdl_path)
        best_acc = cur_acc
        print 'save best module at {}'.format(mdl_path)
    net.train()

