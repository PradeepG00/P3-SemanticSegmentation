from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import sys
import time
import datetime

import numpy as np
import torch
import torchvision.utils as vutils
import tqdm
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from utils.config import AgricultureConfiguration
from utils.data.preprocess import prepare_gt, TRAIN_ROOT, VAL_ROOT
from utils.data.visual import get_visualize, colorize_mask
from utils.metrics.loss import ACWLoss
from utils.metrics.lr import init_params_lr
from core.net import get_model

#####################################
# Setup Logging
#####################################
import logging

from utils.metrics.optimizer import Lookahead
from utils.metrics.validate import AverageMeter, evaluate

logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

model_name = "rx101"
log_path = "./logs/{0}/{1}.log".format(f"/{model_name}", f"{model_name}-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}")
log_dir = f"./logs/{model_name}"
if os.path.exists(log_dir):
    print("Saving log files to:", log_dir)
else:
    print("Creating log directory:", log_dir)
    os.mkdir(log_dir)

fileHandler = logging.FileHandler(
    log_path)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

#####################################
# Training Configuration
#####################################
cudnn.benchmark = True

# the specific meta-data / config of the metrics's training session should
# be stored in either a JSON or YAML format to ease loading
train_args = AgricultureConfiguration(net_name='MSCG-Rx101',
                                      data='Agriculture',
                                      optimizer="SGD",
                                      bands_list=['NIR', 'RGB'],
                                      kf=0, k_folder=0,
                                      note='reproduce'
                                      )
train_args.input_size = [512, 512]
train_args.scale_rate = 1.  # 256./512.  # 448.0/512.0 #1.0/1.0
train_args.val_size = [512, 512]
train_args.node_size = (32, 32)
train_args.train_batch = 6  # 3
train_args.val_batch = 6  # 3, TODO: pretty positive 7 is fine for a 2080TI which has the same MEM SIZE as a 1080TI

train_args.lr = 2.18e-4 / np.sqrt(3)
train_args.weight_decay = 2e-5

train_args.lr_decay = 0.9
train_args.max_iter = 1e8

train_args.snapshot = ''
train_args.print_freq = 100
train_args.save_pred = True # default False

# output training configuration to a text file
train_args.write2txt()
# output training metrics to tensorboard directory
tb_dir = os.path.join(train_args.save_path, 'tblog')
logging.debug("Saving tensorboard results to: {}".format(tb_dir))
writer = SummaryWriter(tb_dir)
visualize, restore = get_visualize(train_args)


# Remember to use num_workers=0 when creating the DataBunch.
def random_seed(seed_value: int, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def train_rx101():
    try:
        prepare_gt(VAL_ROOT)
        prepare_gt(TRAIN_ROOT)
        random_seed(train_args.seeds)
        train_args.write2txt()
        net = get_model(name=train_args.model_name,
                        classes=train_args.nb_classes,
                        node_size=train_args.node_size)
        print(os.path.join(train_args.save_path, 'tblog'))
        save_path = os.path.join(train_args.save_path, 'tblog')
        logging.debug(save_path)
        # checkpoint_path = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_10_loss_1.62912_acc_0.75860_acc-cls_0.54120_mean-iu_0.36020_fwavacc_0.61867_f1_0.50060_lr_0.0001175102.pth"
        checkpoint_path = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_7_loss_1.26578_acc_0.77763_acc-cls_0.53562_mean-iu_0.40502_fwavacc_0.64379_f1_0.54641_lr_0.0001217217.pth"
        net, start_epoch = train_args.resume_train(
            net,
            # checkpoint_path=checkpoint_path
        )
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(0)), strict=False)

        torch.cuda.set_device(0)
        net.cuda()
        net.train()

        train_set, val_set = train_args.get_dataset()
        train_loader = DataLoader(dataset=train_set, batch_size=train_args.train_batch, num_workers=0, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=train_args.val_batch, num_workers=0)

        criterion = ACWLoss().cuda()

        params = init_params_lr(net, train_args)

        # first train with Adam for around 10 epoch, then manually change to SGD
        # to continue the rest train, Note: need resume train from the saved snapshot
        if train_args.optimizer == "adam":
            base_optimizer = optim.Adam(params, amsgrad=True)
        elif train_args.optimizer == "sgd":
            base_optimizer = optim.SGD(params, momentum=train_args.momentum, nesterov=True)
        optimizer = Lookahead(base_optimizer, k=6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60, 1.18e-6)

        new_ep = 7
        print(checkpoint_path is not None and new_ep > 0)
        if checkpoint_path is not None and new_ep > 0:
            logging.debug(f"Resuming using model at: {str(checkpoint_path)}\nstarting at epoch: {new_ep}")
        while True:
            # setup timer for training benchmarking
            start_time = time.time()
            # setup loss metrics
            train_main_loss = AverageMeter()
            aux_train_loss = AverageMeter()
            cls_train_loss = AverageMeter()
            # setup hyperparams
            start_lr = train_args.lr
            train_args.lr = optimizer.param_groups[0]['lr']
            # configure steps
            num_iter = len(train_loader)
            curr_iter = ((start_epoch + new_ep) - 1) * num_iter
            print('---curr_iter: {}, num_iter per epoch: {}---'.format(curr_iter, num_iter))

            for i, (inputs, labels) in enumerate(train_loader):
                sys.stdout.flush()
                # train using GPU
                inputs, labels = inputs.cuda(), labels.cuda(),
                N = inputs.size(0) * inputs.size(2) * inputs.size(3)
                optimizer.zero_grad()
                outputs, cost = net(inputs)  # predict

                main_loss = criterion(outputs, labels)

                loss = main_loss + cost

                loss.backward()
                optimizer.step()
                lr_scheduler.step(epoch=(start_epoch + new_ep))

                train_main_loss.update(main_loss.item(), N)
                aux_train_loss.update(cost.item(), inputs.size(0))

                curr_iter += 1
                writer.add_scalar('main_loss', train_main_loss.avg, curr_iter)
                writer.add_scalar('aux_loss', aux_train_loss.avg, curr_iter)
                # writer.add_scalar('cls_loss', cls_train_loss.avg, curr_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_iter)

                if (i + 1) % train_args.print_freq == 0:
                    new_time = time.time()

                    print('[epoch %d], [iter %d / %d], [loss %.5f, aux %.5f, cls %.5f], [lr %.10f], [time %.3f]' %
                          (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg, aux_train_loss.avg,
                           cls_train_loss.avg,
                           optimizer.param_groups[0]['lr'], new_time - start_time))
                    logging.debug(
                        '[epoch %d], [iter %d / %d], [loss %.5f, aux %.5f, cls %.5f], [lr %.10f], [time %.3f]' %
                        (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg, aux_train_loss.avg,
                         cls_train_loss.avg,
                         optimizer.param_groups[0]['lr'], new_time - start_time))

                    start_time = new_time

            validate(net, val_set, val_loader, criterion, optimizer, start_epoch + new_ep, new_ep)
            end_time = time.time()
            logging.debug(f"training time of epoch-{new_ep}: {end_time - start_time}s")
            new_ep += 1
    except Exception as e:
        # TODO: add clearing out the collected arrays if there is failure
        # TODO: display place of writing the metrics checkpoints
        # TODO: display place of writing the tensorboard logs
        # TODO: display place of writing the
        logging.debug(e)


def validate(net, val_set, val_loader, criterion, optimizer, epoch, new_ep):
    # TODO: why aggregate? is it bad practice or is there purpose?
    net_name = "rx101"
    logging.debug(f"evaluating {net_name} on validation set -- epoch {epoch}")
    net.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []
    i = 0  # DEBUG
    with torch.no_grad():
        for vi, (inputs, gts) in tqdm.tqdm(enumerate(val_loader)):
            logging.debug(f"aggregate input, prediction, ground-truth -- iteration {i}")
            inputs, gts = inputs.cuda(), gts.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            outputs = net(inputs)

            val_loss.update(criterion(outputs, gts).item(), N)
            # val_loss.update(criterion(gts, outputs).item(), N)
            if random.random() > train_args.save_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze(0).cpu())

            gts_all.append(gts.data.squeeze(0).cpu().numpy())
            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            predictions_all.append(predictions)
            i += 1
            logging.debug(f"inputs {len(inputs_all)}, ground-truths {len(gts_all)}")

    update_checkpoint(net, optimizer, epoch, new_ep, val_loss,
                      inputs_all, gts_all, predictions_all)

    net.train()
    return val_loss, inputs_all, gts_all, predictions_all


# TODO: is this somethign that can be multiprocessed ie does not require sequential processing?
def update_checkpoint(net, optimizer, epoch, new_ep, val_loss,
                      inputs_all, gts_all, predictions_all):
    avg_loss = val_loss.avg
    logging.debug("update_checkpoint: evaluating predictions against ground-truths")
    acc, acc_cls, mean_iu, fwavacc, f1 = evaluate(predictions_all, gts_all, train_args.nb_classes)

    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('f1_score', f1, epoch)

    logging.debug("update_checkpoint: updating best record")
    updated = train_args.update_best_record(epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1)

    # save best record and snapshot parameters
    val_visual = []

    snapshot_name = train_args.model_name \
                    + "-" + 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_' \
                            '%.5f_f1_%.5f_lr_%.10f' % (
                        epoch, avg_loss, acc, acc_cls,
                        mean_iu, fwavacc, f1,
                        optimizer.param_groups[0]['lr']
                    )
    logging.debug("checkpointing metrics at: {}".format(os.path.join(train_args.save_path, snapshot_name + '.pth')))
    torch.save(net.state_dict(),
               os.path.join(train_args.save_path, snapshot_name + '.pth'))
    if updated or (train_args.best_record['val_loss'] > avg_loss):
        logging.debug("checkpointing metrics at: {}".format(os.path.join(train_args.save_path, snapshot_name + '.pth')))
        torch.save(net.state_dict(),
                   os.path.join(train_args.save_path, snapshot_name + '.pth'))
        # train_args.update_best_record(epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1)
    if train_args.save_pred:
        if updated:
            # or (new_ep % 5 == 0):
            val_visual = visual_checkpoint(epoch, new_ep, inputs_all, gts_all, predictions_all)

    if len(val_visual) > 0:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)


def visual_checkpoint(epoch, new_ep, inputs_all, gts_all, predictions_all):
    val_visual = []
    if train_args.save_pred:
        save_dir = os.path.join(train_args.save_path, str(epoch) + '_' + str(new_ep))
        check_mkdir(save_dir)

    logging.debug("saving visuals of checkpoint metrics at:", save_dir)
    for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        if data[0] is None:
            continue

        if train_args.val_batch == 1:
            input_pil = restore(data[0][0:3, :, :])
            gt_pil = colorize_mask(data[1], train_args.palette)
            predictions_pil = colorize_mask(data[2], train_args.palette)
        else:
            input_pil = restore(data[0][0][0:3, :, :])  # only for the first 3 bands
            # input_pil = restore(data[0][0])
            gt_pil = colorize_mask(data[1][0], train_args.palette)
            predictions_pil = colorize_mask(data[2][0], train_args.palette)

        # if train_args['val_save_to_img_file']:
        if train_args.save_pred:
            logging.debug("saving prediction to: {}".format(os.path.join(save_dir, '%d_prediction.png' % idx)))
            input_pil.save(os.path.join(save_dir, '%d_input.png' % idx))
            predictions_pil.save(os.path.join(save_dir, '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(save_dir, '%d_gt.png' % idx))

        val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                           visualize(predictions_pil.convert('RGB'))])
    return val_visual


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    train_rx101()
