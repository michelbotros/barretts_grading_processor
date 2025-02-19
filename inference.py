"""
The following is a simple algorithm template that matches the configuration for your algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from pathlib import Path
from glob import glob
import numpy as np
from wholeslidedata import WholeSlideImage
from skimage.measure import label, regionprops
from PIL import Image
import torch
from preprocessing import get_preprocessing
from load_models import Ensemble
import segmentation_models_pytorch as smp
from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMonochromeMaskWriter

INPUT_PATH = Path('/input/images/he-staining')
OUTPUT_PATH = Path('/output/images/barretts-esophagus-dysplasia-heatmap')
RESOURCE_PATH = Path("resources")
DOWNSAMPLING_FACTOR = 4
TILE_SIZE = 512
STEP_SIZE = 256 // 2


def run():
    # start with setting the device
    print("=+=" * 10)
    _show_torch_cuda_info()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    print("=+=" * 10)

    verify_output_folder()

    # read wsi from input path
    wsi_in = load_image(
        location=INPUT_PATH / "",
    )

    print('Input WSI')
    print(f'wsi path: {wsi_in.path}')
    print(f'level count: {wsi_in.level_count}')
    print(f'spacings: {wsi_in.spacings}')
    print(f'shapes:  {wsi_in.shapes}')
    print(f'downsamplings: {wsi_in.downsamplings}')
    print("=+=" * 10)

    # Extract rois from the image
    rois, bboxes = extract_rois_slide(wsi_in, min_size=50 * 10 ** 6)
    print('Starting dysplasia segmentation algorithm'.format(len(rois)))
    print('Left with {} ROIs'.format(len(rois)))

    # Create empty mask equal to the size of the image
    shape_1mpp = wsi_in.shapes[2]
    print("shape at 1mpp: {}".format(shape_1mpp))
    input_spacing = wsi_in.spacings[0]
    print("input_spacing: {}".format(input_spacing))
    output = np.zeros(shape_1mpp, dtype=np.float64)
    output_spacing = input_spacing * DOWNSAMPLING_FACTOR

    # load the models
    exp_dir = 'resources/Ensemble_m5_UNet++_CE_IN/'
    print('Loading models from: {}'.format(exp_dir))
    preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn('efficientnet-b4', 'imagenet'))
    print("=+=" * 10)
    ensemble_m5_CE_IN = Ensemble(exp_dir, device=device, m=5)

    # loop over the rois
    for i, (roi, bb) in enumerate(zip(rois, bboxes)):
        # get the dysplasia heatmap
        print('Segmenting ROI {} with shape: {}'.format(i + 1, roi.shape))
        tile_gen = TileGenerator(image=roi, step_size=STEP_SIZE, tile_size=TILE_SIZE)
        segmentation, counts = extract_segmentation(model=ensemble_m5_CE_IN,
                                                    generator=tile_gen,
                                                    preprocessing=preprocessing,
                                                    device=device)

        y_hat_dys = segmentation[:, :, 2] + segmentation[:, :, 3]
        print('Segmentation shape: {}'.format(y_hat_dys.shape))

        # put the one roi in at the right place
        bb = bb // DOWNSAMPLING_FACTOR
        x_0, y_0, x_1, y_1 = bb
        print('Bounding box: {}'.format(bb))
        output[x_0:x_1, y_0: y_1] = y_hat_dys.T

    print('Segmentation finished')
    print("=+=" * 10)
    # write to output tiff
    outfile = OUTPUT_PATH / "output.tif"
    write_array_as_tif_file(output_file=outfile, output=output, output_spacing=output_spacing)

    # check output
    wsi_out = WholeSlideImage(path=outfile, backend='openslide')
    print("=+=" * 10)
    print('output WSI')
    print(f'wsi path: {wsi_out.path}')
    print(f'level count: {wsi_out.level_count}')
    print(f'spacings: {wsi_out.spacings}')
    print(f'shapes:  {wsi_out.shapes}')
    print(f'downsamplings: {wsi_out.downsamplings}')

    return 0


def verify_output_folder():

    # Check if the directory exists
    if not os.path.exists(OUTPUT_PATH):
        try:
            os.makedirs(OUTPUT_PATH)
            print(f"Directory created: {OUTPUT_PATH}")
        except PermissionError:
            print(f"Permission denied: Cannot create directory at {OUTPUT_PATH}")
    else:
        print(f"Directory already exists: {OUTPUT_PATH}")

    return 0


def write_array_as_tif_file(*, output_file, output, output_spacing):
    """
    """
    dimensions = output.shape
    written = False

    # create writers
    print("Setting up writers, writing to: {}".format(output_file))
    segmentation_writer = WholeSlideMonochromeMaskWriter()
    segmentation_writer.write(path=Path(output_file),
                              spacing=output_spacing,
                              dimensions=dimensions,
                              tile_shape=(TILE_SIZE, TILE_SIZE))

    print("Writing image...")
    for y in range(0, dimensions[1], TILE_SIZE):
        for x in range(0, dimensions[0], TILE_SIZE):

            # get the tile, put onto 0-255 scale
            tile = output[x: x + TILE_SIZE, y: y + TILE_SIZE] * 255

            # skip the last non-fitting tile
            if tile.shape == (TILE_SIZE, TILE_SIZE):
                segmentation_writer.write_tile(tile.T, coordinates=(int(x), int(y)))
                written = True

    if not written:
        raise ValueError(f"No values have been written to {output_file}")

    print('Closing')
    segmentation_writer.save()
    print('Writing finished')


def load_image(*, location):
    # Use wholeslidedata to read a file
    input_files = (
            glob(str(location / "*.tif"))
            + glob(str(location / "*.tiff"))
    )
    # open the WSI
    wsi = WholeSlideImage(input_files[0], backend='openslide')
    return wsi


def mask_to_bbox(mask):
    bboxes = []
    lbl = label(mask)
    props = regionprops(lbl)

    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])

    return bboxes


def extract_rois_slide(wsi, min_size=100 * 10 ** 6, spacing=1.0):
    # open slide at spacing 8.0
    slide_8 = wsi.get_slide(spacing=8.0)

    # to gray
    image_pil = Image.fromarray(slide_8)
    image_gray = np.array(image_pil.convert('L'))
    segmentation = np.where(image_gray < 240, 255, 0)
    bboxes = np.array(mask_to_bbox(segmentation)) * 8 * DOWNSAMPLING_FACTOR  # at 0.25 spacing
    bboxes_filtered = []

    rois = []
    for bb in bboxes:

        x_1, y_1, x_2, y_2 = bb
        top_left = np.array((x_1, y_1))
        bottom_right = np.array((x_2, y_2))

        # get the centre
        diff = bottom_right - top_left
        centre = top_left + (diff / 2)
        x, y = int(centre[0]), int(centre[1])
        width, height = int(diff[0]), int(diff[1])

        # extract roi
        roi = wsi.get_patch(x, y, width // DOWNSAMPLING_FACTOR, height // DOWNSAMPLING_FACTOR, spacing=spacing)

        # discard if too small
        if roi.size > min_size:
            rois.append(roi)
            bboxes_filtered.append(bb)
        else:
            print('Removing roi with size: {} at {}'.format(roi.size, (x, y)))

    return rois, bboxes_filtered


def _show_torch_cuda_info():
    import torch
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")


class TileGenerator:
    """Generates tiles for Numpy images
    """

    def __init__(self, image, step_size, tile_size):

        self.image = image
        self.tile_size = tile_size
        self.step_size = step_size

    def get_generator(self):
        img = self.image
        width, height = img.shape[0], img.shape[1]
        x_tiles = int(np.floor(width / self.step_size))
        y_tiles = int(np.floor(height / self.step_size))

        for y in range(y_tiles):
            for x in range(x_tiles):
                x_coord = int(np.round(x * self.step_size))
                y_coord = int(np.round(y * self.step_size))
                tile = img[x_coord: x_coord + self.tile_size, y_coord: y_coord + self.tile_size]
                centre_coord = (x_coord, y_coord)

                # remove when doesn't fit
                if tile.shape == (self.tile_size, self.tile_size, 3):
                    yield tile, centre_coord


def extract_segmentation(model, generator, preprocessing, device, n_classes=4):
    """

    Args:
            model:
            generator:
            preprocessing:
            device:
            n_classes:

    Returns:
            segmentation: (H, W, C_out)

    """
    tile_size = generator.tile_size
    h, w = generator.image.shape[0], generator.image.shape[1]
    probability_map = np.zeros((h, w, n_classes))  # (H, W, nr_classes)
    prediction_counts = np.zeros((h, w))  # (H, W)
    generator = generator.get_generator()

    with torch.no_grad():
        for idx, (x_np, loc) in enumerate(generator):
            # get location
            x_coord, y_coord = loc

            # pre-process and put on device
            x = preprocessing(image=np.expand_dims(x_np, axis=0))['image'].to(device)
            y = torch.zeros_like(x)

            # forward
            y_hat = np.transpose(model.forward(x, y).squeeze(), (1, 2, 0))

            # (1) put the prediction in the prob map
            probability_map[x_coord: x_coord + tile_size, y_coord: y_coord + tile_size] += y_hat

            # (2) put the count in the count map
            prediction_counts[x_coord: x_coord + tile_size, y_coord: y_coord + tile_size] += 1

    prediction_counts = np.where(prediction_counts == 0, 1, prediction_counts)
    final_probability_map = probability_map / prediction_counts[:, :, None]

    return final_probability_map, prediction_counts


if __name__ == "__main__":
    raise SystemExit(run())
