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
from tqdm import tqdm
from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMonochromeMaskWriter

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():

    # start with setting the device

    print("=+=" * 10)
    _show_torch_cuda_info()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    print("=+=" * 10)

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
    rois, bboxes = extract_rois_slide(wsi_in, min_size=50*10**6)
    print('Starting dysplasia segmentation algorithm'.format(len(rois)))
    print('Found {} ROIs'.format(len(rois)))

    # Create empty mask equal to the size of the image
    shape_1mpp = wsi_in.shapes[2]
    print("shape at 1mpp: {}".format(shape_1mpp))
    input_spacing = wsi_in.spacings[0]
    print("input_spacing: {}".format(input_spacing))
    output = np.zeros(shape_1mpp, dtype=np.float64)
    output_spacing = input_spacing * 4

    # settings originally: tile_size=512, step_size=256
    tile_size = 512
    step_size = 256

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
        tile_gen = TileGenerator(image=roi, step_size=step_size, tile_size=tile_size)
        segmentation, counts = extract_segmentation(model=ensemble_m5_CE_IN,
                                                    generator=tile_gen,
                                                    preprocessing=preprocessing,
                                                    device=device)
        y_hat_dys = segmentation[:, :, 2] + segmentation[:, :, 3]
        print('Segmentation shape: {}'.format(y_hat_dys.shape))

        # put the one roi in at the right place
        bb = bb // 4
        x_0, y_0, x_1, y_1 = bb
        # print('Bounding box: {}'.format(bb))
        output[y_0:y_1, x_0: x_1] = y_hat_dys

    print('Segmentation finished')
    print("=+=" * 10)
    # write to output tiff
    outfile = OUTPUT_PATH / "images" / "output.tif"
    write_array_as_tif_file(location=outfile, array=output, output_spacing=output_spacing)

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


def write_array_as_tif_file(*, location, array, output_spacing):
    """
    """
    dimensions = array.shape
    tile_size = 512

    # create writers
    print("Setting up writers")
    segmentation_writer = WholeSlideMonochromeMaskWriter()
    segmentation_writer.write(path=location,
                              spacing=output_spacing,
                              dimensions=array.shape,
                              tile_shape=(tile_size, tile_size))

    print("Writing image...")
    # loop over image and get tiles To-Do: do with Generator
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):

            # Extract the tile, handling edge cases where the tile goes beyond the image boundaries
            tile = np.zeros((tile_size, tile_size), dtype=array.dtype)
            tile_x_end = min(x + tile_size, dimensions[0])
            tile_y_end = min(y + tile_size, dimensions[1])

            # Copy over the valid region
            tile[: tile_x_end - x, : tile_y_end - y] = array[x:tile_x_end, y:tile_y_end] * 255
            print('Tile range: {}{}'.format(np.min(tile), np.max(tile)))

            # Write the tile using segmentation_writer
            segmentation_writer.write_tile(tile, coordinates=(int(x), int(y)))
    print()
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


def extract_rois_slide(wsi, min_size=100*10**6, spacing=1.0):

    # open slide at spacing 8.0
    slide_8 = wsi.get_slide(spacing=8.0)

    # to gray
    image_pil = Image.fromarray(slide_8)
    image_gray = np.array(image_pil.convert('L'))
    segmentation = np.where(image_gray < 240, 255, 0)
    bboxes = np.array(mask_to_bbox(segmentation)) * 8 * 4  # at 0.25 spacing
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
        roi = wsi.get_patch(x, y, width // 4, height // 4, spacing=spacing)

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
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
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
        for idx, (x_np, loc) in enumerate(tqdm(generator)):

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
