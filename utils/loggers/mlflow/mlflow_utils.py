"""Utilities and tools for tracking runs with MLflow."""

import os
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import LOGGER, colorstr

MLFLOW_PARENT_ENV_VAR = 'MLFLOW_PARENT_RUN_ID'


class MlflowLogger:
    """Log training run, artifacts, parameters, and metrics to Mlflow.

    This logger expects that Mlflow is setup by the user.
    """
    disable_log_model = True  # TODO: Requires properly specified code paths and conda env

    def __init__(self, opt: Namespace) -> None:
        """Initializes the MlflowLogger

        Args:
            opt (Namespace): Commandline arguments for this run
        """
        import mlflow  # load optional dependency lazily
        self.mlflow = mlflow
        self.prefix = colorstr("Mlflow: ")
        parent_run_id = os.environ.get(MLFLOW_PARENT_ENV_VAR)
        self.parent_run = None
        if parent_run_id is not None:
            mlflow.start_run(run_id=parent_run_id)
        self.active_run = mlflow.start_run(nested=parent_run_id is not None)
        with Path(opt.save_dir, 'mlflow_run_id.txt').open('w+') as f:
            f.write(self.active_run.info.run_id)
        LOGGER.info(f"{self.prefix}Using run_id({self.active_run.info.run_id})")
        self.setup(opt)

    def setup(self, opt: Namespace) -> None:
        self.model_name = Path(opt.weights).stem
        try:
            self.client = self.mlflow.tracking.MlflowClient()
            run = self.client.get_run(run_id=self.active_run.info.run_id)
            logged_params = run.data.params
            remaining_params = {k: v for k, v in vars(opt).items() if k not in logged_params}
            self.log_params(remaining_params)
        except Exception:
            LOGGER.exception(f"{self.prefix}Could not log remaining parameters.")
        self.log_metrics(vars(opt), is_param=True)
        self.log_artifacts(Path(opt.hyp))

    def log_artifacts(self, artifact: Path) -> None:
        """Member function to log artifacts (either directory or single item).

        Args:
            artifact (Path): File or folder to be logged
        """
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        if artifact.is_dir():
            self.mlflow.log_artifacts(f"{artifact.resolve()}/", artifact_path=str(artifact.stem))
        else:
            self.mlflow.log_artifact(str(artifact.resolve()))

    def log_model(self, model_path) -> None:
        """Member function to log model as an Mlflow model.

        Args:
            model (nn.Module): The pytorch model
        """
        if not self.disable_log_model:
            model = torch.hub.load(repo_or_dir=str(ROOT.resolve()),
                                   model="custom",
                                   path=str(model_path),
                                   source="local")
            self.mlflow.pytorch.log_model(model, artifact_path=self.model_name, code_paths=[ROOT.resolve()])

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        Mlflow doesn't have mutable parameters and so this function is used
        only to log initial training parameters.

        Args:
            params (Dict[str, Any]): Parameters as dict
        """
        self.mlflow.log_params(params=_safe_keys(params))

    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None, is_param: bool = False) -> None:
        """Log metrics.
        Mlflow requires metrics to be floats.

        Args:
            metrics (Dict[str, float]): Dictionary with metric names and values
            epoch (int, optional): Training epoch. Defaults to None.
            is_param (bool, optional): Set it to True to log keys with a prefix "params/". Defaults to False.
        """
        prefix = "param/" if is_param else ""
        metrics_dict = {
            f"{prefix}{k}": float(v)
            for k, v in metrics.items() if (isinstance(v, float) or isinstance(v, int))}
        self.mlflow.log_metrics(metrics=_safe_keys(metrics_dict), step=epoch)

    def finish_run(self) -> None:
        """Member function to end mlflow run.
        """
        self.mlflow.end_run()


safe_re = re.compile(r"[^a-zA-Z0-9_\-\.\s/]")


def _safe_keys(logs: Dict[str, Any]) -> str:
    """Make keys safe to log to MLflow"""
    return {safe_re.sub("_", k): v for k, v in logs.items()}
