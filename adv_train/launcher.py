# To facilitate running the same code on different machine or cluster.
# This should abstract this by taking a config as input and simply running on the machines specified by the config.
# By default the config should loach locally
# It should support submitit if installed
# It should be implementable as a subclass or could directly launch a function.

import argparse
from ast import arg
from omegaconf import OmegaConf
from enum import Enum
from itertools import product


def get_arg_products(arg_dict: dict) -> list:
    """
    Process a dict of array arguments into a list of keyword arguments of the product of all
    combinations for use in slurm arrays, i.e.:

        x = get_arg_products({'a': [0, 1], 'b': [2, 3], 'c': 4})

        >> x = [{'a': 0, 'b': 2, 'c': 4},
        >>      {'a': 0, 'b': 3, 'c': 4},
        >>      {'a': 1, 'b': 2, 'c': 4},
        >>      {'a': 1, 'b': 3, 'c': 4}]

    """
    if arg_dict is None:
        return []
    listify = [v if isinstance(v, list) else [v] for v in arg_dict.values()]
    product_args = [dict(zip(arg_dict, p)) for p in product(*listify)]
    return product_args


class SlurmPartition(Enum):
    LOCAL = "local"
    DEVFAIR = "devfair"
    LEARNFAIR = "learnfair"
    MILA_LONG = "mila-long"
    MILA_MAIN = "mila-main"
    MILA_UNKILLABLE = "mila-unkillable"

    @staticmethod
    def load(name):
        filename = "./configs/slurm/%s.yml" % name.value
        return OmegaConf.load(filename)


class Launcher:
    @classmethod
    def add_arguments(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--config",
            default=None,
            type=str,
            help="Optional yaml config file specifying default arguments (can be overwritten with "
            "command line arguments)",
        )
        parser.add_argument(
            "--slurm",
            default=SlurmPartition.LOCAL,
            type=SlurmPartition,
            choices=SlurmPartition,
            help="Select slurm partition; specify config in the relevant config file!",
        )
        return parser

    @classmethod
    def parse_args_with_config(cls, parser: argparse.ArgumentParser) -> argparse.Namespace:
        args = parser.parse_args()

        # set arguments from config file
        if args.config is not None:
            config = OmegaConf.load(args.config)
            parser.set_defaults(**config)

        # overwrite args from command line arguments
        args = parser.parse_args()
        return args

    def __init__(self, args):
        self.args = args
        self.__dict__.update(vars(args))
        self.record = None

        if self.slurm is SlurmPartition.LOCAL:
            self.executor = None
        else:
            slurm_config = SlurmPartition.load(self.slurm)
            self.executor = self.create_slurm_executor(slurm_config)

    @staticmethod
    def create_slurm_executor(slurm_config):
        import submitit

        nb_gpus = slurm_config.get("gpus_per_node", 1)
        mem_by_gpu = slurm_config.get("mem_by_gpu", 60)
        log_folder = slurm_config["log_folder"]

        executor = submitit.AutoExecutor(folder=log_folder)

        # TODO: commented out stuff that was breaking my submission - fix this later
        executor.update_parameters(
            slurm_partition=slurm_config.get("partition", ""),
            slurm_comment=slurm_config.get("comment", ""),
            # slurm_constraint=slurm_config.get("gpu_type", ""),
            # slurm_time=slurm_config.get("time_in_min", 30),
            slurm_gres=slurm_config.get("slurm_gres", None),
            timeout_min=slurm_config.get("time_in_min", 30),
            # nodes=slurm_config.get("nodes", 1),
            cpus_per_task=slurm_config.get("cpus_per_task", 10),
            # tasks_per_node=nb_gpus,
            # gpus_per_node=nb_gpus,
            mem_gb=mem_by_gpu * nb_gpus,
            slurm_array_parallelism=slurm_config.get("slurm_array_parallelism", None),
        )
        return executor

    def launch():
        raise NotImplementedError

    def load(self, record):
        self.record = record

    def update_args(self, args):
        self.args = args
        self.__dict__.update(vars(args))

    def run(self, array_args: dict = None):
        product_args = get_arg_products(array_args)

        # local job
        if self.executor is None:
            # sweep over jobs
            if len(product_args) > 0:
                for args in product_args:
                    print(f"Launching job with args: {args}")
                    self.launch(**args)
            # single job
            else:
                self.launch()

        # slurm job
        else:
            # batch job via job arrays
            if len(product_args) > 0:
                jobs = []
                with self.executor.batch():
                    for args in product_args:
                        job = self.executor.submit(self.launch, **args)
                        print(f"Launched batch job: {(str(job.job_id))}, args: {args}")
                        jobs.append(job)
            # single job
            else:
                job = self.executor.submit(self.launch)
                print("Launched job: %s" % (str(job.job_id)))
