#!/usr/bin/env python3
import os
import platform
from pathlib import Path
import tempfile
from typing import Optional
from PIL import Image

import click
import coremltools as ct
import torch
from coremltools.models.neural_network import NeuralNetworkBuilder
from mlflow import artifacts as mlflow_artifacts

from models.experimental import attempt_load
from models.yolo import Detect
from utils.general import LOGGER, check_img_size, colorstr, file_size


@click.command('Export CoreML Model')
@click.option('--weights', type=str, help='Path or mlflow uri of model weights.')
@click.option('--output-dir', type=Path, help='Path of directory to store exported model.')
def main(**kwargs):
    export_coreml_with_nms(**kwargs)


def export_coreml_with_nms(weights: str, output_dir: Optional[Path] = None, image_size=(640, 640), batch_size=1, quantize=False, half=False) -> Path:
    prefix = colorstr('CoreML:')
    mac_capabilities = platform.system() == 'Darwin'
    if not mac_capabilities:
        LOGGER.warning(f'{prefix} Not running on macOS. Quanization and model testing are not supported.')

    # Load PyTorch model
    weights_path = Path(weights if os.path.exists(weights) else mlflow_artifacts.download_artifacts(weights))
    device = 'cpu'
    model = attempt_load(weights_path, device=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Checks
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'
    gs = int(max(model.stride))  # grid size (max stride)
    image_size *= 2 if len(image_size) == 1 else 1  # expand
    assert [check_img_size(x, gs) for x in image_size]  # verify img_size are gs-multiples

    # Input
    im = torch.zeros(batch_size, 3, *image_size).to(device)

    # Update model
    model.eval()  # training mode = no Detect() layer grid construction
    for _, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False # Not compatible
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {weights_path} with output shape {shape} ({file_size(weights_path):.1f} MB)")

    # Export
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)
    file = output_dir / weights_path.with_suffix('.mlmodel').name
    LOGGER.info(f'{prefix} starting export with coremltools {ct.__version__}...')

    ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
    ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
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
                         output_names=['raw_coordinates', 'raw_confidence', 'unknown_confidence'],
                         axis=-1, split_sizes=(4,1,1))

    builder.add_nms(name='nonMaximumSupression',
                    input_names=['raw_coordinates', 'raw_confidence'],
                    output_names=['boxes', 'confidence', 'boxIndices', 'boxNum'],
                    iou_threshold=0.5,
                    score_threshold=0.6,
                    max_boxes=10)

    builder.spec.description.output.add()
    builder.set_output(['boxes', 'confidence'], [(1,10,4), (1,10,1)])
    ct.utils.convert_double_to_float_multiarray_type(builder.spec)
    ct_model = ct.models.MLModel(builder.spec)

    # Test forward pass
    if mac_capabilities:
        img = Image.new('RGB', image_size)
        output = ct_model.predict({'image': img})
        assert 'boxes' and 'confidence' in output
        assert output['boxes'].shape == (1, 10, 4)
        assert output['confidence'].shape == (1, 10, 1)

    ct_model.save(file)

    LOGGER.info(f'{prefix} export success, saved as {file} ({file_size(file):.1f} MB)')
    return file

if __name__ == "__main__":
    main()
