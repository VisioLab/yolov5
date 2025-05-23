#!/usr/bin/env python3
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import click
import coremltools as ct
import mlflow
import torch
from coremltools.models.neural_network import NeuralNetworkBuilder
from mlflow import artifacts as mlflow_artifacts
from PIL import Image

from models.experimental import attempt_load
from models.yolo import Detect
from utils.general import LOGGER, check_img_size, colorstr, file_size, logging_context


@dataclass
class Context:
    model: Any
    sample_img: torch.Tensor
    out_shape: Tuple[int, int]
    weights_path: Path
    output: Path


@click.group('Export')
@click.option('--weights',
              required=True,
              type=str,
              help=('Path or mlflow uri of model weights, uri form: '
                    '"runs:/RUN_ID/path/to/artifact" or "models:/MODEL_NAME/STAGE"'))
@click.option('--output', type=Path, help='Path to store converted model.')
@click.option('--upload', help='log as mlflow artifact, ensure that `MLFLOW_RUN_ID` env var is set', is_flag=True)
@click.pass_context
def cli(ctx, weights, output: Path, upload=False, batch_size=1, image_size=(640, 640)):
    weights_path = Path(weights if os.path.exists(weights) else mlflow_artifacts.download_artifacts(weights))
    model, sample_img, out_shape = initialize_model(weights, batch_size, image_size)

    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {weights_path} with output shape {out_shape} ({file_size(weights_path):.1f} MB)"
    )
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    ctx.obj = Context(model, sample_img, out_shape, weights_path, output)


@cli.result_callback()
def upload(file, **kwargs):
    if kwargs.get('upload'):
        # if `MLFLOW_RUN_ID` is not set, a new run will be created
        with mlflow.start_run():
            mlflow.log_artifact(str(file))


@cli.command()
@click.pass_obj
def coreml(obj: Context):
    _, model = export_coreml_with_nms(obj.model, obj.sample_img)
    model.save(obj.output)
    LOGGER.info(f'Succesully exported model as {obj.output}')
    return obj.output


@cli.command()
@click.pass_obj
def torchscript(obj: Context):
    ts = export_torchscript_model(obj.model, obj.sample_img)
    torch.jit.save(ts, obj.output)  # type: ignore
    LOGGER.info(f'Succesully exported model as {obj.output}')
    return obj.output


@torch.no_grad()
def initialize_model(weights: str, batch_size: int, image_size: Tuple[int, int]):
    # Load PyTorch model
    device = 'cpu'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Checks
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'  # type: ignore
    gs = int(max(model.stride))  # grid size (max stride) # type: ignore
    image_size *= 2 if len(image_size) == 1 else 1  # expand
    assert [check_img_size(x, gs) for x in image_size]  # verify img_size are gs-multiples

    # Input
    im = torch.zeros(batch_size, 3, *image_size).to(device)

    # Update model
    model.eval()  # training mode = no Detect() layer grid construction
    for _, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False  # Not compatible
            m.export = True

    model(im)  # dry runs
    y = model(im)
    shape = tuple(y[0].shape)  # model output shape
    return model, im, shape


def export_torchscript_model(model, sample_img):
    return torch.jit.trace(model, sample_img, strict=False)  # TorchScript model


def export_coreml_with_nms(model,
                           sample_img: torch.Tensor,
                           quantize: bool = False,
                           half: bool = False) -> tuple[Any, ct.models.MLModel]:
    prefix = colorstr('CoreML:')
    mac_capabilities = platform.system() == 'Darwin'
    if not mac_capabilities:
        LOGGER.warning(f'{prefix} Not running on macOS. Quanization and model testing are not supported.')

    ts = export_torchscript_model(model, sample_img)

    LOGGER.info(f'{prefix} starting export with coremltools {ct.__version__}...')

    with logging_context(logging.WARN):
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType('image', shape=sample_img.shape, scale=1 / 255, bias=[0, 0, 0])],
            convert_to="neuralnetwork",
        )
    bits, mode = (8, 'kmeans_lut') if quantize else (16, 'linear') if half else (32, None)
    if bits < 32:
        if mac_capabilities:
            assert mode is not None
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        else:
            print(f'{prefix} quantization only supported on macOS, skipping...')

    # Add NMS layer
    LOGGER.info(f'{prefix} Adding NMS layer to converted model.')
    builder = NeuralNetworkBuilder(spec=ct_model.get_spec())

    [raw_output] = builder.spec.description.output
    builder.add_split_nd(name='split_raw',
                         input_name=raw_output.name,
                         output_names=['raw_coordinates', 'raw_objectness_score', 'raw_confidence'],
                         axis=-1,
                         split_sizes=(4, 1, 1))

    # we only use the objectness score, the confidence is meaningless because it's always 1
    builder.add_nms(name='nonMaximumSupression',
                    input_names=['raw_coordinates', 'raw_objectness_score'],
                    output_names=['boxes', 'confidence', 'boxIndices', 'boxNum'],
                    iou_threshold=0.5,
                    score_threshold=0.6,
                    max_boxes=10)

    builder.spec.description.output.add()
    builder.set_output(['boxes', 'confidence'], [(1, 10, 4), (1, 10, 1)])
    ct.utils.convert_double_to_float_multiarray_type(builder.spec)
    ct_model = ct.models.MLModel(builder.spec)

    # Test forward pass
    if mac_capabilities:
        img = Image.new('RGB', sample_img.shape[-2:])
        output = ct_model.predict({'image': img})
        assert 'boxes' and 'confidence' in output
        assert output['boxes'].shape == (1, 10, 4)
        assert output['confidence'].shape == (1, 10, 1)

    return ts, ct_model


def save_ts_coreml(path: Path, imgsz: int = 640):
    model_load, sample_img, _ = initialize_model(str(path), 1, image_size=(imgsz, imgsz))
    ts_model, coreml_model = export_coreml_with_nms(model_load, sample_img)
    torch.jit.save(ts_model, path.with_suffix('.ts'))
    coreml_model.save(path.with_suffix('.mlmodel'))


if __name__ == "__main__":
    cli()
