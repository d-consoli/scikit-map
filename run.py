from skmap.modeler import RFRegressor, RFRegressorTrees
from skmap.catalog import DataCatalog
from skmap.tiled_data import TiledData, TiledDataLoader, TiledDataExporter
from skmap.misc import TimeTracker, ttprint
import skmap_bindings as sb
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore", module="sklearn")
import time
import gc

version = '20250212'

YEARS = range(2000, 2013, 2)
# YEARS = range(2012, 2023, 2)
DEPTHS = [0, 20, 50, 100, 200]
QUANTILES = [0.16, 0.84]

BASE_PATH = '/mnt/slurm/jobs/ai4sh_pred'
CATALOG_PATH = f'{BASE_PATH}/eu_soil_prop_v{version}.csv'
MODEL_PATH = f'{BASE_PATH}'
TILES_IDS = f'{BASE_PATH}/tiles_eu.in'
MASK_TEMPLATE_PATH = 'http://192.168.49.30:8333/ai4sh/masks/combined/{tile_id}.tif'
GDAL_OPTS = {'GDAL_HTTP_VERSION': '1.0', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'}
GAIA_ADDRS = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
THREADS = 96
DEPTH_VAR = 'hzn_dep'
RESAMPLING_STRATEGY = "GRA_CubicSpline"

S3_PARAMS = {
    's3_addresses':GAIA_ADDRS,
    's3_access_key':'iwum9G1fEQ920lYV4ol9',
    's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
    's3_prefix':f'tmp-ai4sh-layers/eu_props_v{version}',
}
# S3_PARAMS = None

MODE, MODEL_TYPE, MSF = ('depths_years_quantiles', RFRegressorTrees, '.joblib')
# MODE, MODEL_TYPE, MSF = ('depths_years', RFRegressor, '.so')

# SPATIAL_AGGREGATION = 8
SPATIAL_AGGREGATION = None

spatial_res = f'{30*SPATIAL_AGGREGATION}m' if SPATIAL_AGGREGATION else '30m'
out_files_suffix = f'g_epsg.4326_v{version}'

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

# start_tile = 100
# end_tile = 101
# server_name='ripley'


with open(TILES_IDS, 'r') as file:
    tile_ids = [line.strip() for line in file]
tile_ids = tile_ids[start_tile:end_tile]
base_dir = f'/mnt/{server_name}/ai4sh_pred_v2'


ocd_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.ocd_production_v{version}{MSF}',),
    'expm1':True, 'scale':10, 'nodata':32767, 'dtype':'uint16', 'prop_file_name':'oc_iso.10694.1995.mg.cm3'
}
soc_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.soc_production_v{version}{MSF}',),
    'expm1':True, 'scale':10, 'nodata':32767, 'dtype':'uint16', 'prop_file_name':'oc_iso.10694.1995.wpct'
}
ph_h2o_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.ph.h2o_production_v{version}{MSF}',),
    'expm1':False, 'scale':10, 'nodata':255, 'dtype':'byte', 'prop_file_name':'ph.h2o_iso.10390.2021.index'
}
ph_calc2_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.ph.cacl2_production_v{version}{MSF}',),
    'expm1':False, 'scale':10, 'nodata':255, 'dtype':'byte', 'prop_file_name':'ph.cacl2_iso.10390.2021.index'
}
bd_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.bulk.density.fe_production_v{version}{MSF}',),
    'expm1':True, 'scale':100, 'nodata':32767, 'dtype':'uint16', 'prop_file_name':'bd.core_iso.11272.2017.g.cm3'
}
extr_k_params = {
    'model':MODEL_TYPE(model_path=f'{MODEL_PATH}/model_rf.extractable.k_production_v{version}{MSF}',),
    'expm1':True, 'scale':1, 'nodata':32767, 'dtype':'uint16', 'prop_file_name':'k.ext_usda.nrcs.mg.kg'
}

models_params = [
    ocd_params,
    soc_params,
    ph_h2o_params,
    ph_calc2_params,
    bd_params,
    extr_k_params
]

properties_features = {f for params in models_params for f in params['model'].model_covs}
catalog = DataCatalog.create_catalog(catalog_def=CATALOG_PATH, years=YEARS, base_path=GAIA_ADDRS, verbose=False)
YEARS_srt = [str(y) for y in YEARS]
catalog.query(properties_features, YEARS_srt)
properties_data = TiledDataLoader(catalog, MASK_TEMPLATE_PATH, SPATIAL_AGGREGATION, RESAMPLING_STRATEGY, verbose=False)

export_data = TiledDataExporter(spatial_res=spatial_res, s3_params=S3_PARAMS,
                                mode=MODE, years=YEARS, depths=DEPTHS, quantiles=QUANTILES)

for tile_id in tile_ids:
    print("--------------------------------------------------------------")
    
    export_data.tile_id = tile_id
    if all(export_data.check_all_exported(params['prop_file_name'], out_files_suffix) for params in models_params):
        ttprint(f"All properties for tile {tile_id} already computed, skipping")
        continue

    with TimeTracker(f" o Reading data for tile {tile_id}", False):
        properties_data.load_tile_data(tile_id)
        if properties_data.n_pixels_valid == 0:
            ttprint("No pixels to predict in this tile, skipping")
            continue
        properties_data.convert_nan_to_median()
        properties_data.convert_nan_to_value(0.0)
        
    with TimeTracker(f" o Processing tile {tile_id}", False):
        for params in models_params:
            if export_data.check_all_exported(params['prop_file_name'], out_files_suffix):
                ttprint(f"Property {params['prop_file_name']} for tile {tile_id} already computed, skipping")
                continue
            with TimeTracker(f"   # Modeling {params['prop_file_name']}", False):
                properties_model:MODEL_TYPE = params['model']
                with TimeTracker(f"     - Getting predictions", False):
                    pred_depths = []
                    for depth in DEPTHS:
                        with TimeTracker(f"       - Depth {depth}", False):
                            properties_data.fill_otf_constant(DEPTH_VAR, depth)
                            pred_depths += [properties_model.predict(properties_data)]
            with TimeTracker(f"   # Deriving statistics", False):
                if MODE == 'depths_years_quantiles':
                    export_data.derive_block_quantiles_and_mean(pred_depths, params['expm1'])
                elif MODE == 'depths_years':
                    export_data.derive_block_mean(pred_depths, params['expm1'])
                del pred_depths
                gc.collect()
            with TimeTracker(f"   # Exporting files", False):
                export_data.export_files(params['prop_file_name'], out_files_suffix,
                                         params['nodata'], properties_data.mask_path,
                                         params['dtype'], properties_data.get_pixels_valid_idx(1),
                                         write_folder=base_dir, scaling=params['scale'])
                del export_data.array
                gc.collect()
    properties_data.__exit__(None,None,None)
    print("--------------------------------------------------------------")

