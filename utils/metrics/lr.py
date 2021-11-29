from __future__ import print_function, division
import math


def init_params_lr(net, opt):
    """

    :param net:
    :param opt:
    :return:
    """
    bias_params = []
    unbiased_params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                bias_params.append(value)
            else:
                unbiased_params.append(value)
    params = [
        {"params": unbiased_params, "lr": opt.lr, "weight_decay": opt.weight_decay},
        {"params": bias_params, "lr": opt.lr * 2.0, "weight_decay": 0},
    ]
    return params


def lr_poly(base_lr, iteration, max_iterations, power):
    """

    :param base_lr:
    :param iteration:
    :param max_iterations:
    :param power:
    :return:
    """
    return base_lr * ((1 - float(iteration) / max_iterations) ** (power))


def adjust_learning_rate(optimizer, i_iter, opt):
    """

    :param optimizer:
    :param i_iter:
    :param opt:
    :return:
    """
    cur_lr = optimizer.param_groups[0]["lr"]
    # lr = lr_poly(opt.lr, i_iter, opt.max_iter, opt.lr_decay)
    lr = lr_poly(cur_lr, i_iter, opt.max_iter, opt.lr_decay)
    optimizer.param_groups[0]["lr"] = lr  # weights
    optimizer.param_groups[0]["weight_decay"] = opt.weight_decay
    optimizer.param_groups[1]["lr"] = lr * 2.0  # bias
    return lr


def lr_cos(base_lr, iteration, max_iterations):
    """

    :param base_lr:
    :param iteration:
    :param max_iterations:
    :return:
    """
    return base_lr * (1 + math.cos(math.pi * iteration / max_iterations)) / 2.0


def adjust_initial_rate(optimizer, i_iter, opt, model="cos"):
    """

    :param optimizer:
    :param i_iter:
    :param opt:
    :param model:
    :return:
    """
    # lr = lr_poly(opt.lr, i_iter, opt.max_iter, opt.lr_decay)
    if model == "poly":
        lr = lr_poly(
            optimizer.param_groups[0]["initial_lr"], i_iter, opt.max_iter, opt.lr_decay
        )
    else:
        lr = lr_cos(optimizer.param_groups[0]["initial_lr"], i_iter, opt.max_iter)
    optimizer.param_groups[0]["lr"] = lr  # weights
    # optimizer.param_groups[0]['weight_decay'] = opt.weight_decay
    optimizer.param_groups[1]["lr"] = lr * 2.0  # bias
    optimizer.param_groups[0]["initial_lr"] = lr
    return lr
