#!/bin/bash

tile="$1"
letters=( {A..Z} {a..z} )
files=""
calc_expr=""
i=0

while read -r filename; do
  letter=${letters[$i]}
  path="http://192.168.49.30:8333/tmp-global-soil/soil_types_v20250403_tifs/${tile}/${filename}"
  files+=" -${letter}=${path}"
  if [ $i -eq 0 ]; then
    calc_expr="$letter"
  else
    calc_expr="${calc_expr} + $letter"
  fi
  ((i++))
done < haploxerolls.in

gdal_calc.py $files --calc="$calc_expr" \
  --outfile=/mnt/ripley/gen_cog/organic_soils/haploxerolls_sum_${tile}.tif \
  --NoDataValue=255 \
  --co TILED=YES \
  --co BIGTIFF=YES \
  --co COMPRESS=DEFLATE \
  --co PREDICTOR=2 \
  --co BLOCKXSIZE=2048 \
  --co BLOCKYSIZE=2048 \
  --co NUM_THREADS=8 \
  --co SPARSE_OK=TRUE

mc cp /mnt/ripley/gen_cog/organic_soils/haploxerolls_sum_${tile}.tif gaia/tmp-global-soil/haploxerolls_soils_v20250403/haploxerolls_sum_${tile}.tif
rm /mnt/ripley/gen_cog/organic_soils/haploxerolls_sum_${tile}.tif