# To facilitate running the same code on different machine or cluster.
# This should abstract this by taking a config as input and simply running on the machines specified by the config.
# By default the config should loach locally
# It should support submitit if installed
# It should be implementable as a subclass or could directly launch a function.

import argparse
from omegaconf import OmegaConf
from enum import Enum


class SlurmPartition(Enum):
    LOCAL = "local"
    DEVFAIR = "devfair"
    LEARNFAIR = "learnfair"
    MILA_LONG = "mila-long"
    MILA_MAIN = "mila-main"
    MILA_UNKILLABLE = "mila-unkillable"

    @staticmethod
    def load(name):
        filename = "./configs/%s.yml"%name.value
        return OmegaConf.load(filename)
        

class Launcher:
    @classmethod
    def add_arguments(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--slurm", default=SlurmPartition.LOCAL, type=SlurmPartition, choices=SlurmPartition)

        return parser

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
        executor.update_parameters(
            slurm_partition=slurm_config.get("partition", ""),
            slurm_comment=slurm_config.get("comment", ""),
            slurm_constraint=slurm_config.get("gpu_type", ""),
            slurm_time=slurm_config.get("time_in_min", 30),
            timeout_min=slurm_config.get("time_in_min", 30),
            nodes=slurm_config.get("nodes", 1),
            cpus_per_task=slurm_config.get("cpus_per_task", 10),
            tasks_per_node=nb_gpus,
            gpus_per_node=nb_gpus,
            mem_gb=mem_by_gpu * nb_gpus,
            slurm_array_parallelism=slurm_config.get("slurm_array_parallelism", 100),
        )
        
        return executor    

    def launch():
        raise NotImplementedError

    def load(self, record):
        self.record = record

    def update_args(self, args):
        self.args = args
        self.__dict__.update(vars(args))

    def run(self):
        if self.executor is None:
            self.launch()
        else:
            job = self.executor.submit(self.launch)
            print("Launched job: %s"%(str(job.job_id)))






