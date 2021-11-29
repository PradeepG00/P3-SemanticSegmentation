# TODO: implement a similar system to that of ultralytic's yolov5 or other ML CLIs
import click

MODELS = [
    "Rx50",
    "Rx101"
]


@click.group()
def cli():
    # click.echo("Hello World!")
    pass


@cli.command("train", short_help="Run TRAIN for a MODEL for an input DATASET using specified PARAMETERS")
@click.option(
    "-m",
    "--metrics",
    type=click.Choice(choices=MODELS, case_sensitive=False), required=True
)
@click.option(
    "-dp", "--dataset-path", help="Path to DATASET", required=True,
    type=str,
    default=None,
)
@click.option(
    "-pp",
    "--parameters-path",
    type=str,
    default=None,
    help="Path to YAML file containing hyper-parameters",
    required=True,
)
@click.option(
    "-g",
    "--gpu",
    type=bool,
    default=True,
    help="Flag to train using GPU"
)
@click.option(
    "-ng",
    "--n-gpus",
    type=int,
    default=False,
    help="Specification of the NUMBER of GPUs to use to parallelize data during training")
@click.option(
    "-d",
    "--debug",
    type=bool,
    default=True,
    help="Run training process with DEBUG statements and logging"
)
@click.option(
    "-v",
    "--verbose",
    type=bool,
    default=True,
    help="Run training process with VERBOSE message output to stdout"
)
@click.option(
    "-sp",
    "--save-prediction",
    type=bool,
    default=True,
    help="Flag to SAVE PREDICTION during training process"
)
# @tracer
def train(model: str, dataset_path: str, parameters_path: str, gpu: bool, n_gpus: int, debug: bool, verbose: bool):
    # TODO: display training config
    #   -
    print(model, dataset_path, parameters_path, gpu, n_gpus, debug)
    # click.echo(metrics, dataset, parameters)


# TODO: tbd whether necessary or not
# TODO: consider ability to modify the log path with the default being in the package root `./logs`
# TODO: consider ability to modify the checkpointing path with the default being in the package root `./checkpoints`
#   this will need handling to create the necessary folders in the specified directory path
@cli.command("config")
def config():
    pass
