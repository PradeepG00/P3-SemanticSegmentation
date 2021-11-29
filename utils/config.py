import datetime
import os

import numpy as np
import torch
import yaml
from pathlib import Path

from utils.data.dataset import AgricultureDataset
from utils.data.preprocess import land_classes, palette_vsl, split_train_val_test_sets


class AgricultureConfiguration(object):
    root_path = Path("/home/hanz/github/P3-SemanticSegmentation")
    model_name = 'none'
    optimizer = "adam"
    # data set parameters
    dataset = 'Agriculture'
    bands = ['NIR', 'RGB']
    loader = AgricultureDataset
    labels = land_classes
    nb_classes = len(land_classes)
    palette = palette_vsl  # palette_vsl
    weights = []

    k_folder = 0
    k = 0
    input_size = [512, 512]
    scale_rate = 1.0 / 1.0
    val_size = [512, 512]  # [1224, 1224]#[2248, 2248]
    train_samples = 12901
    val_samples = 1200
    train_batch = 7
    val_batch = 7
    # train_batch = 1
    # val_batch = 1

    # flag of mean_std normalization to [-1, 1]
    pre_norm = False
    seeds = 69278  # random seed

    # training hyper parameters
    lr = 1.5e-4 / np.sqrt(3.0)
    lr_decay = 0.9
    max_iter = 1e8

    # l2 regularization factor, increasing or decreasing to fight over-fitting
    weight_decay = 2e-5
    momentum = 0.9

    # check point parameters
    checkpoint_path = root_path / "checkpoints" / optimizer
    snapshot = ''
    print_freq = 100
    # print_freq = 5
    save_pred = False
    save_rate = 0.1
    best_record = {}

    def __init__(self, net_name=model_name, data=dataset, bands_list=bands, kf=1, k_folder=5, note='',
                 optimizer="adam"):
        """Class handling training configuration and tensorboard operations

        :param net_name:
        :param data:
        :param bands_list:
        :param kf:
        :param k_folder:
        :param note:
        """
        self.model_name = net_name
        self.dataset = data
        self.bands = bands_list
        self.k = kf
        self.k_folder = k_folder
        self.suffix_note = note

        check_mkdir(self.checkpoint_path)
        check_mkdir(os.path.join(self.checkpoint_path, self.model_name))

        band_str = '-'.join(self.bands)
        if self.k_folder is not None:
            sub_dir = self.dataset + '_' + band_str + '_kf-' + str(self.k_folder) + '-' + str(self.k)
        else:
            sub_dir = self.dataset + '_' + band_str
        if note != '':
            sub_dir += '-'
            sub_dir += note

        check_mkdir(os.path.join(self.checkpoint_path, self.model_name, sub_dir))
        self.save_path = os.path.join(self.checkpoint_path, self.model_name, sub_dir)

    def get_file_list(self):
        return split_train_val_test_sets(name=self.dataset, bands=self.bands, KF=self.k_folder, k=self.k,
                                         seeds=self.seeds)

    def get_dataset(self):
        train_dict, val_dict, test_dict = self.get_file_list()
        # split_train_val_test_sets(name=self.dataset,KF=None, k=self.k,seeds=self.seeds)
        train_set = self.loader(mode='train', file_lists=train_dict, pre_norm=self.pre_norm,
                                num_samples=self.train_samples, win_size=self.input_size, scale=self.scale_rate)
        val_set = self.loader(mode='val', file_lists=val_dict, pre_norm=self.pre_norm,
                              num_samples=self.val_samples, win_size=self.val_size, scale=self.scale_rate)
        return train_set, val_set

    def resume_train(self, net, checkpoint_path: str = None, use_gpu: bool=True):
        """Function for loading in a metrics for resuming training
z
        :param net:
        :param checkpoint_path:
        :return:
        """
        # check if path exists
        if checkpoint_path:
            print("Loading metrics from checkpoint:", checkpoint_path)
            if use_gpu:
                net.load_state_dict(torch.load(checkpoint_path,map_location=torch.device(0)), strict=False)
            else:
                net.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu')), strict=False)

        if len(self.snapshot) == 0:
            curr_epoch = 1
            self.best_record = {'epoch': 0, 'val_loss': 0, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0,
                                'f1': 0}
        else:
            print('training resumes from ' + self.snapshot)
            # core.load_state_dict(torch.load(self.snapshot))
            net.load_state_dict(torch.load(os.path.join(self.save_path, self.snapshot)))
            split_snapshot = self.snapshot.split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            self.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11]),
                                'f1': float(split_snapshot[13])}
        return net, curr_epoch

    def print_best_record(self):
        print(
            '[best_ %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
                self.best_record['epoch'],
                self.best_record['val_loss'], self.best_record['acc'],
                self.best_record['acc_cls'],
                self.best_record['mean_iu'], self.best_record['fwavacc'], self.best_record['f1']
            ))

    def update_best_record(self, epoch, val_loss,
                           acc, acc_cls, mean_iu,
                           fwavacc, f1):
        print('----------------------------------------------------------------------------------------')
        print('[epoch %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
            epoch, val_loss, acc, acc_cls, mean_iu, fwavacc, f1))
        self.print_best_record()

        print('----------------------------------------------------------------------------------------')
        if mean_iu > self.best_record['mean_iu'] or f1 > self.best_record['f1']:
            self.best_record['epoch'] = epoch
            self.best_record['val_loss'] = val_loss
            self.best_record['acc'] = acc
            self.best_record['acc_cls'] = acc_cls
            self.best_record['mean_iu'] = mean_iu
            self.best_record['fwavacc'] = fwavacc
            self.best_record['f1'] = f1
            return True
        else:
            return False

    def display(self):
        """printout all configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write2txt(self):
        file = open(os.path.join(self.save_path,
                                 str(datetime.datetime.now()) + '.txt'), 'w')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                line = "{:30} {}".format(a, getattr(self, a))
                file.write(line + '\n')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    import re
    import json

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open("../data/config.yml", 'r') as fh:
        config = yaml.load(fh, Loader=loader)
        print(config)
    with open("../data/config.json", 'w') as fh:
        json.dump(config, fh, indent=4)
    pass
