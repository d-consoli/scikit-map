from skmap.modeler import RFRegressorDepths, PredictedDepths
from skmap.catalog import s3_setup, DataCatalog, s3_list_files
from skmap.loader import TiledDataLoader
from skmap.misc import TimeTracker
import os
from skmap.misc import GoogleSheet
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import skmap_bindings as sb
import random
import json
import tl2cgen
warnings.filterwarnings("ignore", module="sklearn")


YEARS = [2000, 2005, 2010, 2015, 2020, 2022]
DEPTHS = [0, 30, 60, 100]
QUANTILES = [0.16, 0.84]

catalog_csv = '/mnt/slurm/jobs/global_soc/landsat_and_co.csv'
TILES_PATH = '/mnt/slurm/jobs/global_soc/ard2_final_status.gpkg'
MODEL_PATH = '/mnt/slurm/jobs/global_soc'
TILES_SHUF = '/mnt/slurm/jobs/global_soc/tiles_shuf2.txt'
MASK_TEMPLATE_PATH = 'http://192.168.49.30:8333/global/tiled.masks/mask_landsat_glad.lc.landmask_glc.desert.ice/{tile_id}.tif'
GDAL_OPTS = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}
S3_PREFIX = '/tmp-gpw/global_soc_v4'
# S3_PREFIX = None

ACCESS_KEY = 'iwum9G1fEQ920lYV4ol9'
SECRET_KEY = 'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0'
THREADS = 96
GAIA_ADDRS = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
GDAL_OPTS = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}
s3_aliases = s3_setup(S3_PREFIX is not None, ACCESS_KEY, SECRET_KEY, GAIA_ADDRS)

resampling_strategy = "GRA_CubicSpline"
convert_nan_to_num = True
spatial_aggregation = True

tiles = gpd.read_file(TILES_PATH)
start_tile=max(int(sys.argv[1]), 0)
end_tile=min(int(sys.argv[2])+1, len(tiles))
with open(TILES_SHUF, 'r') as file:
    shuf = [int(line.strip()) for line in file]
tiles_id = tiles['TILE'][shuf[start_tile:end_tile]].tolist()
server_name=sys.argv[3]
base_dir = f'/mnt/{server_name}/global_soc/tmp_data'
os.makedirs(f'/mnt/{server_name}/global_soc', exist_ok=True)
os.makedirs(base_dir, exist_ok=True)

version = '20250110'

texture1_params = {
        'model':RFRegressorDepths(
            model_name='texture1',
            model_path=f'{MODEL_PATH}/model_rf.texture1_production_v{version}.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1,
        'nodata':255,
        'dtype':'uint8',
        'prop_file_name': 'sand.tot_iso.11277.2020.wpct',
        's3_prefix':S3_PREFIX
    }
 
    # texture2
texture2_params = {
        'model':RFRegressorDepths(
            model_name='texture2',
            model_path=f'{MODEL_PATH}/model_rf.texture2_production_v{version}.joblib',
            model_covs_path=None,
            depth_var='hzn_dep',
            depths=DEPTHS,
            predict_fn=lambda predictor, data: predictor.predict(data)
        ),
        'expm1':True,
        'scale':1,
        'nodata':255,
        'dtype':'uint8',
        'prop_file_name': 'silt.tot_iso.11277.2020.wpct',
        's3_prefix':S3_PREFIX
    }

texture1:RFRegressorDepths = texture1_params['model']
texture2:RFRegressorDepths = texture2_params['model']
# NOTE must have the same numbers of trees
assert(texture1.n_trees == texture2.n_trees)
n_trees = texture1.n_trees

textures_model_params = [texture1_params, texture2_params]
textures_features = {f for params in textures_model_params for f in params['model'].model_features}

catalog = DataCatalog.create_catalog(catalog_def=catalog_csv, years=YEARS, base_path=GAIA_ADDRS)
YEARS_srt = [str(y) for y in YEARS]
catalog.query(textures_features, YEARS_srt)
textures_data = TiledDataLoader(catalog, MASK_TEMPLATE_PATH, spatial_aggregation, resampling_strategy, GDAL_OPTS, THREADS)

catalog.save_json('textures_catalog.json')

        
for tile_id in tiles_id:
    skip_flag = True
    computed_list = s3_list_files(s3_aliases, S3_PREFIX, tile_id)
    for params in [texture1_params, texture2_params]:
        computed_list_prop = [s for s in computed_list if params['prop_file_name'] in s]
        if not(len(computed_list_prop) == \
            (len(QUANTILES) + 1) * (len(DEPTHS) - 1) * (len(YEARS) - 1)):
            skip_flag = False
    if skip_flag:
        print(f"Textures 1 and 2 for tile {tile_id} already computed, skipping...")
        continue
    with TimeTracker(f" - Reading data for tile {tile_id}", False):
        textures_data.load_tile_data(tile_id)
        if textures_data.n_pixels_valid == 0:
            print("No pixels to predict in this tile, skipping")
            continue
        if convert_nan_to_num:
            textures_data.convert_nan(0.0)
        n_years = len(YEARS)
        n_depths = len(DEPTHS)
        n_quant = len(QUANTILES)
        n_years_avg = n_years - 1
        n_depths_avg = n_depths - 1
        n_pix = textures_data.x_size * textures_data.y_size
        n_pix_val = textures_data.n_pixels_valid
        n_trees = texture1.n_trees
        n_textures = 3
        n_files = n_depths_avg * n_years_avg * n_textures * (n_quant + 1)
        # This order is also representing the ordering of the array            
        n_files_dephts = n_years_avg * n_textures * (n_quant + 1) # Offset of the dephts
        n_files_years = n_textures * (n_quant + 1) # Offset of the years
        # Textures order: clay, sand, silt
        out_data_t = np.empty((n_pix_val, n_files), dtype=np.float32)
        write_data_t = np.empty((n_pix, n_files), dtype=np.float32)
        write_data = np.empty((n_files, n_pix), dtype=np.float32)
        nodata = 255

    with TimeTracker(f" - Getting raw tree predictions", False):
        # Get raw trees predictions            
        # [n_depths](n_trees, n_samples)
        pred1_trees = [texture1.predictDepth(textures_data, i) for i in range(n_depths)]
        pred2_trees = [texture2.predictDepth(textures_data, i) for i in range(n_depths)]

    with TimeTracker(f" - Deriving statistics", False):
        # Compute derived statistics
        out_files = []
        for d in range(n_depths_avg):
            for y in range(n_years_avg):
                trees1_avg = np.empty((n_trees, n_pix_val), dtype=np.float32)
                trees2_avg = np.empty((n_trees, n_pix_val), dtype=np.float32)                    
                sb.blocksAverage(trees1_avg, THREADS, pred1_trees[d], pred1_trees[d+1], n_pix_val, y)
                sb.blocksAverage(trees2_avg, THREADS, pred2_trees[d], pred2_trees[d+1], n_pix_val, y)      
                trees1_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
                trees2_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
                mean1 = np.empty((n_pix_val,), dtype=np.float32)
                mean2 = np.empty((n_pix_val,), dtype=np.float32)                    
                sb.transposeArray(trees1_avg, THREADS, trees1_avg_t)
                sb.transposeArray(trees2_avg, THREADS, trees2_avg_t)
                sb.nanMean(trees1_avg_t, THREADS, mean1)
                sb.nanMean(trees2_avg_t, THREADS, mean2)
                if texture1_params['expm1']:
                    np.expm1(mean1, out=mean1)
                    np.expm1(trees1_avg_t, out=trees1_avg_t)
                if texture2_params['expm1']:
                    np.expm1(mean2, out=mean2)                        
                    np.expm1(trees2_avg_t, out=trees2_avg_t)
                clay_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                sand_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                silt_trees = np.empty((n_pix_val, n_trees), dtype=np.float32)
                sb.fitPercentage(clay_trees, THREADS, trees1_avg_t, trees2_avg_t)
                sb.hadamardProduct(sand_trees, THREADS, trees1_avg_t, clay_trees)
                sb.hadamardProduct(silt_trees, THREADS, trees2_avg_t, clay_trees)                
                PERCENTILES = [q*100. for q in QUANTILES]                    
                offset_caly = d * n_files_dephts + y * n_files_years
                offset_sand = d * n_files_dephts + y * n_files_years + (n_quant + 1)
                offset_silt = d * n_files_dephts + y * n_files_years + 2 * (n_quant + 1)
                sb.computePercentiles(clay_trees, THREADS, range(clay_trees.shape[1]), out_data_t, range(offset_caly+1,offset_caly+1+len(PERCENTILES)), PERCENTILES)
                sb.computePercentiles(sand_trees, THREADS, range(sand_trees.shape[1]), out_data_t, range(offset_sand+1,offset_sand+1+len(PERCENTILES)), PERCENTILES)
                sb.computePercentiles(silt_trees, THREADS, range(silt_trees.shape[1]), out_data_t, range(offset_silt+1,offset_silt+1+len(PERCENTILES)), PERCENTILES)
                clay_mean = np.empty((n_pix_val,), dtype=np.float32)
                sand_mean = np.empty((n_pix_val,), dtype=np.float32)
                silt_mean = np.empty((n_pix_val,), dtype=np.float32)                    
                sb.fitPercentage(clay_mean, THREADS, mean1, mean2)
                sb.hadamardProduct(sand_mean, THREADS, mean1, clay_mean)
                sb.hadamardProduct(silt_mean, THREADS, mean2, clay_mean)
                out_data_t[:,offset_caly] = clay_mean
                out_data_t[:,offset_sand] = sand_mean
                out_data_t[:,offset_silt] = silt_mean
                res_str = '240m' if spatial_aggregation else '30m'
                for t in ['clay.tot_iso.11277.2020.wpct', 'sand.tot_iso.11277.2020.wpct', 'silt.tot_iso.11277.2020.wpct']:
                    out_files.append(f'{t}_m_{res_str}_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_g_epsg.4326_v{version}')
                    for q in QUANTILES:
                        formatted_p = 'p0' if (q == 0) else ('p100' if (q == 1) else str(q).replace('0.','p'))
                        out_files.append(f'{t}_{formatted_p}_{res_str}_b{DEPTHS[d]}cm..{DEPTHS[d+1]}cm_{YEARS[y]}0101_{YEARS[y+1]}1231_g_epsg.4326_v{version}')

    with TimeTracker(f" - Saving results", False):
        sb.offsetAndScale(out_data_t, THREADS, 0.5, 1.)
        sb.fillArray(write_data_t, THREADS, nodata)
        sb.expandArrayRows(out_data_t, THREADS, write_data_t, textures_data.get_pixels_valid_idx(1))
        sb.transposeArray(write_data_t, THREADS, write_data)
        tile_dir = base_dir + f'/{tile_id}'
        os.makedirs(tile_dir, exist_ok=True)
        write_idx = range(len(out_files))
        compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
        s3_out = None
        if S3_PREFIX is not None:
            s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{S3_PREFIX}/{textures_data.tile_id}' for _ in range(len(out_files))]
        sb.writeByteData(write_data, THREADS, GDAL_OPTS, [textures_data.mask_path for _ in range(len(out_files))], tile_dir, out_files, 
                                         write_idx, 0, 0, textures_data.x_size, textures_data.y_size, int(nodata), compress_cmd, s3_out)

        
    if os.path.exists(textures_data.mask_path):
        os.remove(textures_data.mask_path)
        print(f"{textures_data.mask_path} has been deleted.")
    else:
        print(f"{textures_data.mask_path} does not exist.")


        