"""Utilities and tools for tracking runs with MLflow."""

import os
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

import backoff
import mlflow
import requests
import torch

from urllib3.exceptions import MaxRetryError
from google.api_core.exceptions import GoogleAPICallError

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import LOGGER, colorstr

MLFLOW_PARENT_ENV_VAR = 'MLFLOW_PARENT_RUN_ID'

T = TypeVar("T")

retry = backoff.on_exception(
    backoff.expo,
    (mlflow.MlflowException, requests.exceptions.ConnectionError, MaxRetryError, GoogleAPICallError),
    logger=LOGGER,
    max_tries=8,
)


class MlflowLogger:
    """Log training run, artifacts, parameters, and metrics to Mlflow.

    This logger expects that Mlflow is setup by the user.
    """

    def __init__(self, opt: Namespace) -> None:
        """Initializes the MlflowLogger

        Args:
            opt (Namespace): Commandline arguments for this run
        """
        self.prefix = colorstr("Mlflow: ")
        parent_run_id = os.environ.get(MLFLOW_PARENT_ENV_VAR)
        self.parent_run = None
        if parent_run_id is not None:
            mlflow.start_run(run_id=parent_run_id)
        run_id_path = Path(opt.save_dir, "mlflow_run_id.txt")
        run_id = None
        if run_id_path.exists():
            run_id = run_id_path.read_text()
        self.active_run = mlflow.start_run(run_id=run_id, nested=parent_run_id is not None)
        if not run_id_path.exists():
            run_id_path.write_text(self.active_run.info.run_id)
        LOGGER.info(f"{self.prefix}Using run_id({self.active_run.info.run_id})")
        self.setup(opt)

    def setup(self, opt: Namespace) -> None:
        self.model_name = Path(opt.weights).stem
        try:
            self.client = mlflow.tracking.MlflowClient()
            run = self.client.get_run(run_id=self.active_run.info.run_id)
            logged_params = run.data.params
            remaining_params = {k: v for k, v in vars(opt).items() if k not in logged_params}
            self.log_params(remaining_params)
        except Exception:
            LOGGER.exception(f"{self.prefix}Could not log remaining parameters.")
        self.log_metrics(vars(opt), is_param=True)
        self.log_artifacts(Path(opt.hyp))

    @retry
    def log_artifacts(self, artifact: Path) -> None:
        """Member function to log artifacts (either directory or single item).

        Args:
            artifact (Path): File or folder to be logged
        """
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        if artifact.is_dir():
            mlflow.log_artifacts(f"{artifact.resolve()}/", artifact_path=str(artifact.stem))
        else:
            mlflow.log_artifact(str(artifact.resolve()))

    @retry
    def log_model(self, model_path) -> None:
        """Member function to log model as an Mlflow model.

        Args:
            model (nn.Module): The pytorch model
        """
        model = torch.hub.load(repo_or_dir=str(ROOT.resolve()), model="custom", path=str(model_path), source="local")
        root = ROOT.resolve()
        mlflow.pytorch.log_model(
            model,
            artifact_path=self.model_name,
            code_paths=[root / "utils", root / "models", root / "detect.py"],
            pip_requirements=str(root / "requirements.txt"),
        )

    @retry
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        Mlflow doesn't have mutable parameters and so this function is used
        only to log initial training parameters.

        Args:
            params (Dict[str, Any]): Parameters as dict
        """
        mlflow.log_params(params=_safe_keys(params))

    @retry
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
        mlflow.log_metrics(metrics=_safe_keys(metrics_dict), step=epoch)

    @retry
    def finish_run(self) -> None:
        """Member function to end mlflow run.
        """
        mlflow.end_run()


safe_re = re.compile(r"[^a-zA-Z0-9_\-\.\s/]")


def _safe_keys(logs: Dict[str, T]) -> Dict[str, T]:
    """Make keys safe to log to MLflow"""
    return {safe_re.sub("_", k): v for k, v in logs.items()}
