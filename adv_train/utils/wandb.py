import argparse
from typing import Union
import logging
import os
import wandb
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "busycalibrating")


class WandB:
    @classmethod
    def add_arguments(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(description="Master parser")

        group = parser.add_argument_group(
            "Weights and Biases",
            description="Arguments to support creating a Weights and Biases run.",
        )

        group.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Name to be assigned to the WandB run. Leave blank for auto-generated names",
        )
        group.add_argument(
            "--wandb_project", 
            type=str, 
            default=None, 
            help="WandB project. Leave empty to disable WandB"
        )
        group.add_argument(
            "--wandb_entity",
            default=DEFAULT_WANDB_ENTITY,
            type=str,
            help="An entity is a username or team name where you're sending runs. This entity "
            "must exist before you can send runs there, so make sure to create your account "
            "or team in the UI before starting to log runs. If you don't specify an entity, "
            "the run will be sent to your default entity, which is usually your username. "
            "Change your default entity in your settings",
        )
        group.add_argument(
            "--wandb_group",
            type=str,
            default=None,
            help="Specify a group to organize individual runs into a larger experiment. "
            "For example, you might be doing cross validation, or you might have "
            "multiple jobs that train and evaluate a model against different test sets. "
            "Group gives you a way to organize runs together into a larger whole, and "
            "you can toggle this on and off in the UI.",
        )
        return parser

    @classmethod
    def parse(cls, args: argparse.ArgumentParser):
        # TODO: there has to be a way to encapsulate this into a super class?
        kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
        )
        return kwargs

    @classmethod
    def exec(cls, entity, project, name=None, group=None, **kwargs) -> Union[wandb.sdk.wandb_run.Run, None]:
        """Passes init information into

        Args:
            entity ([type]): [description]
            project ([type]): [description]
            name ([type], optional): [description]. Defaults to None.
            group ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        if project is not None:
            logger.info(f"Logging to Weights and Biases")
            logger.info(f"Project:\t{project}")
            logger.info(f"Entity:\t{entity}")
            logger.info(f"Name:\t{name}")
            logger.info(f"Group:\t{group}")
            if kwargs is not None:
                logger.info(f"Additional kwargs:\t{kwargs}")

            run = wandb.init(name=name, entity=entity, project=project, group=group, **kwargs)

        else:
            logger.warn(f"--wandb_project is None; not logging to WandB!")
            run = None

        return run
