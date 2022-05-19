import json
import click
from pathlib import Path
from shutil import make_archive
import os
import datetime

from src.experiment import experiments

DATA_DIR = Path.cwd().joinpath('data')
DUMP_DIR = Path.cwd().joinpath('dumps')
SRC_DIR = Path.cwd().joinpath('src')


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--debug/--release', default=False)
@click.pass_context
def main(ctx, config_path):
    with open(config_path) as f:
        config = json.load(f)

    model_name = config['model']['name']
    foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    dump_dir = DUMP_DIR.joinpath(model_name + "_" + foldername)
    os.mkdir(dump_dir)

    ctx.obj["data_dir"] = dump_dir
    ctx.obj["config"] = config
    json.dump(config, open(dump_dir.joinpath("configs.json"), "w"), indent=4)
    make_archive(dump_dir.joinpath("src"), "zip", root_dir=SRC_DIR)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def ate(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments(config, data_dir, num_thread)


if __name__ == '__main__':
    main(obj={})
    # for debugging: you can pass in a desired config without the command-line like:
    # main(['configs/your_config_file.json', 'ate'], obj={})
