import os
import numpy as np
import rasterio
import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image, ImageFile
import cv2
import random
from omegaconf import OmegaConf
import warnings

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def resize_to(img, to, is_mask=False):
    """
    Resize an image or mask to a square target size.
    Parameters: 
        img (np.ndarray) - input image or mask array; 
        to (int) - target output size (to x to); 
        is_mask (bool) - whether the input is a mask (nearest interpolation).
    Returns: 
        np.ndarray - resized image or mask.
    """

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    interp = cv2.INTER_NEAREST

    if img.ndim == 3:  # (H, W, C)
        H, W, c = img.shape
        out = np.zeros((to, to, c), dtype=img.dtype)
        for i in range(c):
            out[:,:,i] = cv2.resize(img[:,:,i].reshape((H,W)), (to, to), interpolation=interp)
        return out

    else:  # mask (H, W)
        return cv2.resize(img, (to, to), interpolation=interp)


def center_crop(img, crop_size):
    """
    Extract a centered square crop from an image or mask.
    Parameters: 
        img (np.ndarray) - input image or mask array; 
        crop_size (int) - size of the square crop.
    Returns: 
        np.ndarray - cropped image or mask.
    """
    if img.ndim == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    cy, cx = H // 2, W // 2
    half = crop_size // 2

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    if img.ndim == 3:
        return img[y1:y2, x1:x2, :]
    else:
        return img[y1:y2, x1:x2]
    

def label_to_mask(label):
    """
    Convert a multi-channel label image into a binary mask.
    Parameters: 
        label (np.ndarray) - input label image with values in [0-255].
    Returns: 
        np.ndarray - binary mask with values {0,1}.
    """
    label = label[...,0]
    label[label != 0] = 1
    return np.astype(label, np.uint8)


def extract_random_sample(img_arr, mask_arr, shift_based_on_size, tile_size):
    """
    Extract a randomly shifted square sample from an image and its corresponding mask.
    Parameters: 
        img_arr (np.ndarray) - source image array; 
        mask_arr (np.ndarray) - corresponding label array; 
        shift_based_on_size (int) - maximum random shift allowed; 
        tile_size (int) - size of the extracted sample.
    Returns: 
        tuple[np.ndarray, np.ndarray] - sampled image tile and its corresponding mask tile.
    """

    H, W = img_arr.shape[0], img_arr.shape[1]
    starting_point_x = (W - tile_size) // 2
    starting_point_y = (H - tile_size) // 2
    minx = max(-shift_based_on_size, -starting_point_x)
    maxx = min(shift_based_on_size, W - starting_point_x - tile_size)
    miny = max(-shift_based_on_size, -starting_point_y)
    maxy = min(shift_based_on_size, H - starting_point_y - tile_size)
    shift_x = random.randint(minx, maxx)
    shift_y = random.randint(miny, maxy)
        
    x0 = starting_point_x - shift_x
    y0 = starting_point_y - shift_y

    tile = img_arr[y0:y0+tile_size, x0:x0+tile_size, :]
    tile_mask = mask_arr[y0:y0+tile_size, x0:x0+tile_size, :]
    return tile, tile_mask


def _process_tile_scale(task):
    """
    Worker function: process one tile at one scale.
    Must be defined at module level for Windows multiprocessing.
    """
    (
        tile_idx,
        img_name,
        img_src,
        masks_src,
        scale,
        id_scale,
        base_squares,
        crop_size,
        sample_size,
        num_samples_per_tile,
        output_dirs,
    ) = task

    src_img = os.path.join(img_src, img_name)
    src_mask = os.path.join(masks_src, img_name)

    # Read and close both raster files.
    with rasterio.open(src_img) as src:
        img_arr = np.moveaxis(src.read(), 0, 2)

    with rasterio.open(src_mask) as src:
        mask_arr = np.moveaxis(src.read(), 0, 2)

    counts = np.zeros((len(base_squares), 2), dtype=np.int64)

    for id_base, base_size in enumerate(base_squares):

        area_img = img_arr.shape[0] * img_arr.shape[1]
        area_sample = base_size ** 2
        num_max_samples = int(round(area_img / area_sample / 2, 0))
        num_samples = min(num_samples_per_tile, num_max_samples)

        if num_samples <= 0:
            continue

        samp_img_arr, samp_label_arr = extract_random_sample(
            img_arr, mask_arr, base_size, crop_size
        )

        for i in range(num_samples):
            filename = (
                f"{os.path.splitext(img_name)[0]}"
                f"_scale_{scale}_base_square_{base_size}_{i}.tif"
            )

            base_dir = output_dirs[id_base]

            src_img_out = os.path.join(base_dir, "images", filename)
            src_label_out = os.path.join(base_dir, "labels", filename)
            src_mask_out = os.path.join(base_dir, "masks", filename)

            cropped_img = center_crop(
                samp_img_arr, int(crop_size * scale)
            )
            resized_img = resize_to(cropped_img, sample_size)

            cropped_label = center_crop(
                samp_label_arr, int(crop_size * scale)
            )
            resized_label = resize_to(cropped_label, sample_size)

            resized_mask = label_to_mask(resized_label)

            compression_options = {
                "compression": "zstd",
                "compressionargs": {"level": 9},
            }

            tiff.imwrite(
                src_img_out, resized_img, **compression_options
            )
            tiff.imwrite(
                src_label_out, resized_label, **compression_options
            )
            tiff.imwrite(
                src_mask_out, resized_mask, **compression_options
            )

            if np.sum(resized_mask[:-1, ...] > 0) == 0:
                counts[id_base, 0] += 1
            else:
                counts[id_base, 1] += 1

    return id_scale, tile_idx, counts


def preprocessing(args):
    """
    Generate segmentation training datasets using multiprocessing.
    """

    # General arguments
    tiles_loc = args.preprocessing.tiles_location
    tiles_img_src = os.path.join(tiles_loc, "images")
    tiles_masks_src = os.path.join(tiles_loc, "masks")
    results_src = args.preprocessing.results_location
    suffixe = args.preprocessing.suffixe

    # Segmentation arguments
    sample_size = int(args.preprocessing.sample_size)
    base_squares = list(args.preprocessing.base_squares)
    scales = list(args.preprocessing.scales)
    crop_size = int(sample_size / min(scales))
    num_samples_per_tile = int(
        args.preprocessing.num_samples_per_tile
    )

    # Number of worker processes; configure this in your YAML if desired.
    assert 0.0 < args.preprocessing.frac_workers <= 1.0
    max_workers = max([1, int(args.preprocessing.frac_workers * os.cpu_count())]) if args.preprocessing.do_multi_proc else 1
    print(f"Processing with {max_workers} workers.")

    assert os.path.isdir(tiles_img_src)
    assert os.path.isdir(tiles_masks_src)

    image_files = {
        x for x in os.listdir(tiles_img_src)
        if os.path.splitext(x)[1].lower() in (".png", ".tif", ".tiff")
    }
    mask_files = set(os.listdir(tiles_masks_src))

    missing_masks = image_files - mask_files
    if missing_masks:
        raise RuntimeError(
            f"Missing masks for images: {sorted(missing_masks)[:10]}"
        )

    os.makedirs(results_src, exist_ok=True)

    list_tiles_img = sorted(image_files)
    if not list_tiles_img:
        raise RuntimeError(f"No images found in {tiles_img_src}")

    print("PRODUCING DATASET FOR SEGMENTATION:")
    dataset_src = os.path.join(results_src, f"dataset_segmenter_{suffixe}")
    dataset_base_src = os.path.join(results_src, f"dataset_segmenter_multi_bases_{suffixe}")
    os.makedirs(dataset_base_src, exist_ok=True)
    os.makedirs(dataset_src, exist_ok=True)

    # Prepare all output directories before starting workers.
    output_dirs_by_scale = {}

    for scale in scales:
        scale_dir = os.path.join(dataset_base_src, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)

        output_dirs = []
        for base_size in base_squares:
            base_dir = os.path.join(
                scale_dir, f"scale_{scale}_base_size_{base_size}"
            )

            for subdir in ("images", "masks", "labels"):
                os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

            output_dirs.append(base_dir)

        output_dirs_by_scale[scale] = output_dirs

    # Counts have shape: (scales, base sizes, empty/occupied).
    count_empties = np.zeros(
        (len(scales), len(base_squares), 2), dtype=np.int64
    )

    # One task per tile and scale.
    tasks = []
    for id_scale, scale in enumerate(scales):
        for tile_idx, img_name in enumerate(list_tiles_img):

            tasks.append((
                tile_idx,
                img_name,
                tiles_img_src,
                tiles_masks_src,
                scale,
                id_scale,
                base_squares,
                crop_size,
                sample_size,
                num_samples_per_tile,
                output_dirs_by_scale[scale],
            ))

    # Multiprocessing: workers return statistics; main process aggregates.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_tile_scale, tasks)

        for id_scale, tile_idx, counts in tqdm(
            results, total=len(tasks), desc="Processing tiles"
        ):
            count_empties[id_scale] += counts

    # Calculate occupied percentage for each scale and base size.
    total_samples = count_empties.sum(axis=2)
    occupied_samples = count_empties[:, :, 1]

    results_frac_arr = np.divide(
        occupied_samples * 100.0,
        total_samples,
        out=np.zeros_like(occupied_samples, dtype=float),
        where=total_samples > 0,
    )

    # Plot occupied fraction.
    linestyles = ["-", "-.", "--", ":", "-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, scale in enumerate(scales):
        ax.plot(
            base_squares,
            results_frac_arr[i],
            label=str(scale),
            linewidth=3,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=1,
            alpha=1.0,
        )

    ax.set_xlabel("base squares [px]")
    ax.set_ylabel("Occupied samples [%]")
    ax.set_ylim([0, 105])
    ax.set_xticks(base_squares)
    ax.legend(title="Scale")
    ax.grid()

    plt.title("Fraction of occupied samples")
    plt.tight_layout()

    saving_loc = os.path.join(
        dataset_base_src, "fraction_of_occupied.png"
    )
    fig.savefig(saving_loc)
    fig.savefig(os.path.splitext(saving_loc)[0] + ".eps", format="eps")
    plt.close(fig)

    print(f"Dataset saved to: {dataset_base_src}")
    print(f"Worker processes used: {max_workers}")


if __name__ == "__main__":
    from time import time
    t0 = time()
    args = OmegaConf.load("./config/preprocessing.yaml")

    preprocessing(args)

    # Show duration of process
    delta_time_loop = time() - t0
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"\n==== PREPROCESSING done in {hours}:{min}:{sec} ====\n")
