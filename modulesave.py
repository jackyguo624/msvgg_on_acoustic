import torch


def load_module(module, path):
        # print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        module.start_epoch = checkpoint['epoch']
        module.best_prec1 = checkpoint['best_prec1']
        device_ids = checkpoint['device_ids']
        first_device = device_ids[0]
        module.device_ids = device_ids
        module.first_device = first_device
        module = torch.nn.DataParallel(module, device_ids=device_ids)
        module.cuda(first_device)
        module.load_state_dict(checkpoint['state_dict'])
#        print("=> loaded checkpoint '{}' (epoch {})"
#              .format(path, checkpoint['epoch']))


def save_module(module, epoch, best_prec1, device_ids,
                path='./checkpoint.cpt'):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': module.state_dict(),
        'best_prec1': best_prec1,
        'device_ids': device_ids
    }, path)
