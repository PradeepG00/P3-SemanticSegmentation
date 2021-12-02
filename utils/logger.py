import os
import logging
import datetime


#####################################
# Setup Logging
# TODO: integrate to be used on each module with master config when running various CLI commands
#####################################
# logging.basicConfig(level=logging.DEBUG)
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# rootLogger = logging.getLogger()
#
# fileHandler = logging.FileHandler(
#     "./logs/{0}/{1}.log".format("./", f"rx50-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"))
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)
#
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
from utils import PROJECT_ROOT


def setup_logger(log_directory: str, model_name: str) -> None:
    """
    Function for setting up the logger for debugging purposes

    :param log_directory:
    :param model_name:
    :return:
    """
    logging.basicConfig(level=logging.DEBUG)
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    log_path = PROJECT_ROOT / "logs/{0}/{1}.log".format(
        f"/{model_name}", f"{model_name}-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"
    )
    log_dir = PROJECT_ROOT / f"logs/{model_name}"
    if os.path.exists(log_dir):
        print("Saving log files to:", log_dir)
    else:
        print("Creating log directory:", log_dir)
        os.mkdir(log_dir)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def tracer(func):
    """
    Decorator to print function call details
    :param func:
    :return:
    """
    # Getting the argument names of the
    # called function
    arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]

    # Getting the Function name of the
    # called function
    f_name = func.__name__

    def inner_func(*args, **kwargs):
        print(f_name, "(", end="")

        # printing the function arguments
        print(
            ", ".join(
                "% s = % r" % entry for entry in zip(arg_names, args[: len(arg_names)])
            ),
            end=", ",
        )

        # Printing the variable length Arguments
        print("args =", list(args[len(arg_names) :]), end=", ")

        # Printing the variable length keyword
        # arguments
        print("kwargs =", kwargs, end="")
        print(")")

    return inner_func
