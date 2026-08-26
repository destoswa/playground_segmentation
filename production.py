import os
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import DBSCAN
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.affinity import affine_transform
from omegaconf import OmegaConf
from time import time
import torch
import tifffile as tiff

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.production_utils import download_tile, produce_with_lower_res, predict_with_batch, geo_transfert, load_best_checkpoint

from transformers import SegformerForSemanticSegmentation

# Clearing warnings
Image.MAX_IMAGE_PIXELS = None
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning
)


def tiles_downloading(
        dest_tiles, 
        downloading_mode, 
        canton=None,
        area=None,
        year=None,
        dest_not_empty='add'
        ):
    tiles_to_download = []
    lst_tiles_src = []
    os.makedirs(dest_tiles, exist_ok=True)
    if len(os.listdir(dest_tiles)) > 0:
        if dest_not_empty == 'replace':
            shutil.rmtree(dest_tiles)
            os.makedirs(dest_tiles, exist_ok=True)
        elif dest_not_empty == 'stop':
            raise PermissionError('The destination already contains files. Empty it or change parameter "dest_not_empty"')
        
    # find tiles to download
    tiles_locs = gpd.read_file("utils/resources/tiles_locs/ch.swisstopo.images-swissimage-dop10.metadata.shp")
    if downloading_mode == 'year':
        tiles_locs = tiles_locs.loc[tiles_locs.datenstand == str(year)]
    ids = tiles_locs.id.values
    E = [x.split('_')[0] for x in ids]
    N = [x.split('_')[1] for x in ids]
    EN = [[int(x), int(y)] for x,y in zip(E,N)]

    if downloading_mode == 'canton':
        cantons = gpd.read_file('utils/resources/swissboundaries/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp')
        if canton not in cantons.NAME.values:
            raise AttributeError(f"The given canton's name is not correct. Please choos between the following: \n {cantons.NAME.values}")
        
        canton_polygons = cantons[cantons.NAME == canton]
        # make sure CRS match
        if tiles_locs.crs != canton_polygons.crs:
            tiles_locs = tiles_locs.to_crs(canton_polygons.crs)
        # spatial join - keep tiles that intersect Vaud
        tiles_vaud = gpd.sjoin(tiles_locs, canton_polygons, how="inner", predicate="intersects")
        tiles_to_download = [[int(x) for x in tile.split('_')] for tile in tiles_vaud.id.to_list()]
        print(f"Processing canton {canton} with {len(tiles_to_download)} tiles:")
    elif downloading_mode == 'area':
        # find area of interest
        Emin = int(area.Emin)
        Emax = int(area.Emax)
        Nmin = int(area.Nmin)
        Nmax = int(area.Nmax)

        tiles_to_download = [x for x in EN if Emin <= x[0] <= Emax and Nmin <= x[1] <= Nmax]

        # Gives information about tiles to be downloaded
        text = f"""
    ({Emin},{Nmax+1}) --- ({Emax+1},{Nmax+1})
        |               |
        |               |
        |               |
    ({Emin},{Nmin}) --- ({Emax+1},{Nmin})
    """
        print(f"Processing following area ({len(tiles_to_download)} tiles):\n{text}")

    elif downloading_mode in ['year', 'full']:
        
        tiles_to_download = EN

        # Gives information about tiles to be downloaded
        if downloading_mode == 'year':
            print(f"Processing data of the year {year} ({len(tiles_to_download)} tiles):")
        else:
            print(f"Processing all of Switzerland ({len(tiles_to_download)})")
    else:
        raise AttributeError("downloader.mode is not a valid value!")
        
    # download tiles
    for _, tile in tqdm(enumerate(tiles_to_download), total=len(tiles_to_download), desc="Downloading"):
        tile_src = download_tile(tile[0], tile[1], dest_tiles)
        if tile_src != None:
            lst_tiles_src.append(tile_src)

    return lst_tiles_src


def prediction(
        src_img, 
        src_inter, 
        src_dest_preds, 
        src_dest_probas, 
        resolutions, 
        model_dir, 
        batch_size=8,
        tile_size=512, 
        stride=256, 
        threshold_proba= 0.5,
        do_save_mask=True,
        do_save_img=True,
        do_save_inter=True,
        do_save_probas=True,
        ):
    
    # predict at each resolution
    images = []
    preds = []
    probas = []

    # load model
    ckpt_path = load_best_checkpoint(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    model.eval()

    for res in resolutions:
        res_img, src_res_img = produce_with_lower_res(src_img, src_inter, res, do_save=do_save_inter, do_show=False)
        _, preds_img, proba_img = predict_with_batch(
            image=res_img, 
            model=model, 
            img_path=src_res_img,
            batch_size=batch_size,
            tile_size=tile_size,
            stride=stride,
            th=threshold_proba, 
            do_show=False,
            do_save=False,
            do_save_mask_as_img=False,
            )
        images.append(res_img)
        preds.append(preds_img)
        probas.append(proba_img)

    # merge different resolutions into one final
    with Image.open(src_img) as original_img:
        W, H = original_img.size
    # original_img = Image.open(src_img)

    # W, H = original_img.size

    final_probas = np.zeros((W,H), dtype=np.float32)

    for proba in probas:
        rescaled_proba = Image.fromarray(proba).resize((W, H), Image.NEAREST)
        final_probas += rescaled_proba

    final_probas /= len(probas)
    final_probas = np.clip(final_probas, 0, 1)

    final_product = np.zeros((H, W), dtype=np.uint8)
    final_product[final_probas >= threshold_proba] = 1
    final_probas = (final_probas * 255).astype(np.uint8)

    # save inter
    if do_save_inter:
        for id_res, res in enumerate(resolutions):
            rescaled_mask = Image.fromarray(preds[id_res]).resize((W, H), Image.NEAREST)
            src_dest_preds_mask = os.path.join(src_inter, os.path.splitext(os.path.basename(src_img))[0] + f'_res_{res}_preds_mask.tif')
            tiff.imwrite(src_dest_preds_mask, rescaled_mask, compression="zstd", compressionargs={"level": 9})

    # creation of final preds
    final_product_rgb = np.zeros((W, H, 3))
    final_product_rgb[np.broadcast_to(final_product[:, :, np.newaxis], final_product_rgb.shape) == 1] = 255

    src_final_preds_mask = os.path.join(src_dest_preds, os.path.splitext(os.path.basename(src_img))[0] + f'_mask.tif')
    src_final_preds_img = os.path.join(src_dest_preds, os.path.splitext(os.path.basename(src_img))[0] + f'_img.tif')
    src_final_probas_mask = os.path.join(src_dest_probas, os.path.splitext(os.path.basename(src_img))[0] + f'_probas.tif')

    if do_save_mask:
        tiff.imwrite(src_final_preds_mask, final_product.astype(np.uint8), compression="zstd", compressionargs={"level": 9})
    if do_save_img:
        tiff.imwrite(src_final_preds_img, final_product_rgb.astype(np.uint8), compression="zstd", compressionargs={"level": 9})
    if do_save_probas:
        tiff.imwrite(src_final_probas_mask, final_probas, compression="zstd", compressionargs={"level": 9})

    return final_product, src_final_preds_mask, src_final_preds_img, src_final_probas_mask#, src_final_probas_img


def clustering(
        img_arr, 
        src_dest, 
        eps, 
        min_samples, 
        min_cluster_size,  
        color_palette, 
        do_save_img=True,
        ):
    # extract coordinates of landslides
    pos_ls = np.argwhere(img_arr)

    mask_clusters = np.zeros(img_arr.shape)

    if len(pos_ls) > 0:
        # create cluster map
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=2).fit(pos_ls)

        # unpack coordinates
        rows = pos_ls[:, 0]
        cols = pos_ls[:, 1]

        # remove noise
        labels = clustering.labels_
        valid = labels != -1
        rows = rows[valid]
        cols = cols[valid]
        labels = labels[valid]
        labels[labels == 0] = np.max(labels) + 1

        # count cluster sizes (FAST)
        unique, counts = np.unique(labels, return_counts=True)

        # find clusters to keep
        keep_clusters = unique[counts >= min_cluster_size]

        # mask of points to keep
        keep_mask = np.isin(labels, keep_clusters)

        # now write ONLY good points to image
        mask_clusters[rows[keep_mask], cols[keep_mask]] = labels[keep_mask]

        # saving image version
        if do_save_img:
            rgb_clusters = np.zeros((mask_clusters.shape[0], mask_clusters.shape[1], 4))
            lst_clusters = set(keep_clusters)
            distinct_colors_rgb8 = [(x, y, z, 255) for [x,y,z] in color_palette]

            for cluster in lst_clusters:
                id_color = cluster % len(distinct_colors_rgb8)
                rgb_clusters[mask_clusters == cluster] = distinct_colors_rgb8[id_color]

    # save results
    src_mask = os.path.splitext(src_dest)[0] + f'_clusters_eps_{eps}_min_samp_{min_samples}_mask.tif'
    src_img = os.path.splitext(src_dest)[0] + f'_clusters_eps_{eps}_min_samp_{min_samples}_img.tif'

    if do_save_img:
        tiff.imwrite(src_img, rgb_clusters.astype(np.uint8), compression="zstd", compressionargs={"level": 9})
    tiff.imwrite(src_mask, mask_clusters.astype(np.uint16), compression="zstd", compressionargs={"level": 9})

    return src_mask, src_img


def vectorize(src_target, src_dest):
    with rasterio.open(src_target) as src:
        mask = src.read(1)
        if np.sum(mask) == 0:
            return
        transform = src.transform
        crs = src.crs

    records = [
        {"geometry": shape(geom), "raster_val": value}
        for geom, value in shapes(mask, transform=transform)
        if value != 0
    ]

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # apply affine transform if original masks are note georeferenced
    if transform == rasterio.transform.IDENTITY and crs is None:
        # Flip Y coordinates to match raster display (y → height - y)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: affine_transform(geom, [1, 0, 0, -1, 0, 0])
        )

    src_polygons = os.path.join(src_dest, os.path.splitext(os.path.basename(src_target))[0] + '_playgrounds.gpkg')
    gdf.to_file(src_polygons, layer='polygons', driver="GPKG")

    # compute centroids
    geometry_points = gdf.geometry.centroid
    gdf_centroids = gpd.GeoDataFrame(gdf, geometry=geometry_points, crs=crs)
    gdf_centroids.to_file(src_polygons, layer='centroids', driver="GPKG")

    return src_polygons


def merge_final_results(src_target, src_output, verbose=False):
    list_files = [os.path.join(src_target, x) for x in os.listdir(src_target) if x.endswith('.gpkg')]
    output_polygons = os.path.join(src_output, "merged_polygons.gpkg")
    output_centroids = os.path.join(src_output, "merged_centroids.gpkg")

    # --------------------------------
    # Load and merge
    # --------------------------------
    polygons_list = []
    centroids_list = []

    for f in list_files:
        gdf_polygons = gpd.read_file(f, layer='polygons')
        gdf_centroids = gpd.read_file(f, layer='centroids')

        # Extract polygons
        poly = gdf_polygons[['geometry']].copy()  # adjust fields as needed
        poly['update_state'] = None
        poly['source'] = os.path.basename(f)
        polygons_list.append(poly)

        # Extract centroids
        cent = gdf_centroids[['geometry']].copy()  # adjust fields as needed
        cent['update_state'] = None
        cent['source'] = os.path.basename(f)
        centroids_list.append(cent)

    # --------------------------------
    # Concatenate
    # --------------------------------
    merged_polygons = gpd.GeoDataFrame(
        pd.concat(polygons_list, ignore_index=True),
        crs=polygons_list[0].crs
    )

    merged_centroids = gpd.GeoDataFrame(
        pd.concat(centroids_list, ignore_index=True),
        crs=centroids_list[0].crs
    )

    # --------------------------------
    # Save
    # --------------------------------
    merged_polygons.to_file(output_polygons, driver="GPKG")
    merged_centroids.to_file(output_centroids, driver="GPKG")
    if verbose:
        print(f"Polygons saved to {output_polygons} — {len(merged_polygons)} features")
        print(f"Centroids saved to {output_centroids} — {len(merged_centroids)} features")


def production(args):
    start_time = time()

    # Load parameters
    DEST_ORIGINAL_TILES = args.data.source
    DEST_NOT_EMPTY = args.downloader.dest_not_empty
    SKIP_AUTO_DOWNLOADING = args.downloader.skip_auto_downloading
    DOWNLOADING_MODE = args.downloader.mode
    CANTON = args.downloader.canton
    AREA = args.downloader.area
    YEAR = args.downloader.year
    DEST_PREDS = args.data.results
    MODEL_DIR = args.predictions.model_dir
    BATCH_SIZE = args.predictions.batch_size
    THRESHOLD_PREDS = args.predictions.threshold_preds
    TILE_SIZE = args.predictions.tile_size
    OVERLAP = args.predictions.overlap
    STRIDE = TILE_SIZE - OVERLAP
    RESOLUTIONS = args.predictions.scales
    KEEP_INTERMED_FILES = args.to_keep.intermed
    KEEP_MASK_BIN = args.to_keep.mask_bin
    KEEP_MASK_IMG = args.to_keep.mask_img
    KEEP_PROBAS = args.to_keep.probas
    KEEP_CLUSTER_MONO = args.to_keep.cluster_mono
    KEEP_CLUSTER_IMG = args.to_keep.cluster_img

    DEST_PREDS = DEST_ORIGINAL_TILES if DEST_PREDS.lower() == 'default' else DEST_PREDS
    dest_inter_dir = os.path.join(DEST_PREDS, '0_inter')
    dest_probas_dir = os.path.join(DEST_PREDS, '1_probas')
    dest_preds_dir = os.path.join(DEST_PREDS, '2_predictions')
    dest_clusters_dir = os.path.join(DEST_PREDS, '3_clusters')
    dest_vectors_dir = os.path.join(DEST_PREDS, '4_vectors')

    # === TILES DOWNLOADING ===
    # =========================
    os.makedirs(DEST_ORIGINAL_TILES, exist_ok=True)
    if not SKIP_AUTO_DOWNLOADING:
        lst_tiles_src = tiles_downloading(
            dest_tiles=DEST_PREDS,
            downloading_mode=DOWNLOADING_MODE,
            canton=CANTON,
            area=AREA,
            year=YEAR,
            dest_not_empty=DEST_NOT_EMPTY
        )
    else:
        img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        lst_tiles_src = [os.path.join(DEST_ORIGINAL_TILES, x) for x in os.listdir(DEST_ORIGINAL_TILES) if os.path.splitext(x)[1].lower() in img_exts]

    if len(lst_tiles_src) == 0:
        print("NO TILE TO PROCESS!")
        return
    
    if KEEP_MASK_BIN or KEEP_MASK_IMG:
        os.makedirs(dest_preds_dir, exist_ok=True)
    if KEEP_PROBAS:
        os.makedirs(dest_probas_dir, exist_ok=True)
    if KEEP_INTERMED_FILES:
        os.makedirs(dest_inter_dir, exist_ok=True)

    os.makedirs(dest_clusters_dir, exist_ok=True)
    os.makedirs(dest_vectors_dir, exist_ok=True)

    for _, src_img in tqdm(enumerate(lst_tiles_src), total=len(lst_tiles_src), desc="Processing tiles"):
        # === PREDICTIONS =====
        # =====================
        pred_mask_arr, src_pred_mask, src_pred_img, src_proba_mask = prediction(
            src_img=src_img,
            src_inter=dest_inter_dir,
            src_dest_preds=dest_preds_dir, 
            src_dest_probas=dest_probas_dir,
            resolutions=RESOLUTIONS, 
            model_dir=MODEL_DIR, 
            batch_size=BATCH_SIZE,
            tile_size=TILE_SIZE,
            stride=STRIDE,
            threshold_proba=THRESHOLD_PREDS, 
            do_save_mask=KEEP_MASK_BIN,
            do_save_img=KEEP_MASK_IMG,
            do_save_inter=KEEP_INTERMED_FILES,
            do_save_probas=KEEP_PROBAS,
            )

        # === VECTORIZATION ===
        # =====================
        EPS = args.vectorization.dbscan_eps
        MIN_SAMPLES = args.vectorization.dbscan_min_samples
        MIN_CLUSTER_SIZE = args.vectorization.min_cluster_size
        SRC_COLOR_PALETTE = args.vectorization.src_color_palette
        with open(SRC_COLOR_PALETTE, 'r') as f:
            color_palette = json.load(f)

        src_clusters_mask, src_clusters_img = clustering(
            img_arr=pred_mask_arr,
            src_dest=os.path.join(dest_clusters_dir, os.path.basename(src_img)),
            eps= EPS, 
            min_samples=MIN_SAMPLES, 
            min_cluster_size=MIN_CLUSTER_SIZE,
            color_palette=color_palette,
            do_save_img=KEEP_CLUSTER_IMG,
            )

        # georeference files
        if KEEP_MASK_BIN:
            geo_transfert(src_img, src_pred_mask, True)
        if KEEP_MASK_IMG:
            geo_transfert(src_img, src_pred_img, True)
        if KEEP_PROBAS:
            geo_transfert(src_img, src_proba_mask, True)
        # if KEEP_CLUSTER_MONO:
        geo_transfert(src_img, src_clusters_mask, True)
        if KEEP_CLUSTER_IMG:
            geo_transfert(src_img, src_clusters_img, True)
        
        if KEEP_INTERMED_FILES:
            tile_name = os.path.splitext(os.path.basename(src_img))[0]
            for src_inter in [os.path.join(dest_inter_dir, x) for x in os.listdir(dest_inter_dir) if tile_name in x]:
                geo_transfert(src_img, src_inter)

        # vectorize if any cluster found
        vectorize(src_clusters_mask, dest_vectors_dir)

        if not KEEP_CLUSTER_MONO:
            os.remove(src_clusters_mask)
    if not os.listdir(dest_clusters_dir):
        os.rmdir(dest_clusters_dir)

    # Merge final results
    if args.vectorization.do_merge_results:
        merge_final_results(
            src_target=dest_vectors_dir, 
            src_output=os.path.dirname(dest_vectors_dir),
            verbose=args.verbose,
            )

    # Show duration of process
    delta_time_loop = time() - start_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"\n==== FINISH! {len(lst_tiles_src)} tiles processed in {hours}:{min}:{sec} ====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    cfg_path = args.config

    if cfg_path != "":
        print("- Producing from argument - ")
        args = OmegaConf.load(cfg_path)
    else:
        print("- Producing from yaml file - ")
        args = OmegaConf.load('config/production.yaml')

    production(args)
