import os
import uuid
from datetime import datetime

from logs.wandb_logger import WandbLogger


def setup_logger(args_cli):
    # exp time and unique_id
    exp_time = datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    unique_id = str(uuid.uuid4())[:4]

    # Get the current date and time
    args_cli.group = "-".join((exp_time, unique_id))
    args_cli.name = "-".join(
        ("Test", args_cli.task, unique_id, "seed:" + str(args_cli.seed))
    )
    args_cli.logdir = os.path.join("logs", args_cli.group)

    default_cfg = vars(args_cli)
    logger = WandbLogger(
        config=default_cfg,
        project=args_cli.project,
        group=args_cli.group,
        name=args_cli.name,
        log_dir=args_cli.logdir,
        log_txt=True,
    )
    logger.save_config(default_cfg, verbose=True)

    return logger
