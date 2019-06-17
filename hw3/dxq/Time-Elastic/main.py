
import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from utils.dataset import DataSet
from TENet.TENet import TENet
from utils.transforms import *
from utils.opts import args

best_prec1 = 0

def run():
    global best_prec1
    check_rootfolders()

    args.store_name = '_'.join(
        ['TENet', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)
    args.store_name = args.snapshot_pref
    model = TENet(args.num_class, args.num_segments, args.modality,
                  base_model=args.arch,
                  consensus_type=args.consensus_type,
                  dropout=args.dropout,
                  img_feature_dim=args.img_feature_dim,
                  partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.pretrain_model:
        if os.path.isfile(args.pretrain_model):
            print(("=> loading checkpoint '{}'".format(args.pretrain_model)))
            checkpoint = torch.load(args.pretrain_model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.test, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.pretrain_model)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['RGBDiff']:
        data_length = 5
    else:
        raise ValueError('Unknown modality: {}'.format(args.modality))

    if args.dataset == "pku":
        prefix = "image_{:05d}.jpg"
    elif args.dataset == "ntu":
        prefix = "image_{:04d}.jpg"
    else:
        # something-something format
        prefix = "{:06d}.jpg"

    train_loader = torch.utils.data.DataLoader(
        DataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[40,60,80,100],
                                                     gamma=0.5)

    if args.test:
        test(val_loader, model)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.snapshot_pref), 'w')
    summary_dir = "summary/{}".format(args.snapshot_pref)
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, epoch, log_training, writer)
        scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, loss_val = validate(val_loader, model, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

            writer.add_scalar("Valid/acc", prec1, epoch)
            writer.add_scalar("Valid/loss", loss_val, epoch)


def train(train_loader, model, optimizer, epoch, log, writer):

    criterion = torch.nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    model.train_inception = True
    model.judge_test = False
    end = time.time()
    n = 0

    for i, (input,  target) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        if args.ada_type != "null":
           all_input, all_label = adversarial(model, input, target)
        else:
            all_input, all_label = input, target

        output, _, _, _ = model(all_input)
        loss = criterion(output, all_label)

        prec1, prec5 = accuracy(output.data, all_label, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5,
                        lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        n = n + 1

    writer.add_scalar("Train/acc", top1.avg, epoch)
    writer.add_scalar("Train/loss", losses.avg, epoch)

def adversarial(model, input, target):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    mse_criterion = torch.nn.MSELoss().cuda()

    # feature extract for adversarial training
    with torch.no_grad():
        _, first_feature, _, second_feature = model(input)

    # generate the first adversarial sample
    adv_input = input.clone()
    adv_input = torch.autograd.Variable(adv_input, requires_grad=True)
    max_optimizer = torch.optim.SGD([{'params': adv_input}], 1.0)

    adv_target, adv_first_feature, _, _ = model(adv_input)
    max_loss = criterion(adv_target, target) - 1 * mse_criterion(adv_first_feature, first_feature)
    max_loss = -max_loss
    max_optimizer.zero_grad()
    max_loss.backward()
    max_optimizer.step()

    # concate first sample to origin input
    all_input = torch.cat((input, adv_input), dim=0)
    all_label = torch.cat((target, target), dim=0)

    # generate the second adversarial sample
    if args.ada_type != "ada":
        adv_input = input.clone()
        adv_input = torch.autograd.Variable(adv_input, requires_grad=True)
        max_optimizer = torch.optim.SGD([{'params': adv_input}], 1.0)

        _, _, adv_target, adv_second_feature = model(adv_input)
        max_loss = criterion(adv_target, target) - 1 * mse_criterion(adv_second_feature,
                                                                     second_feature)
        max_loss = -max_loss
        max_optimizer.zero_grad()
        max_loss.backward()
        max_optimizer.step()

        # concate first sample to origin input
        all_input = torch.cat((all_input, adv_input), dim=0)
        all_label = torch.cat((all_label, target), dim=0)

    return all_input, all_label

def validate(val_loader, model, iter, log):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.train_inception = False
    model.judge_test = True
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            # compute output
            output, _, _, _ = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg, losses.avg


def test(val_loader, model):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.train_inception = False
    model.judge_test = True
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            # compute output
            output, _, _, _ = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))

    decay = 0.1 ** (epoch // 10)
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    run()
