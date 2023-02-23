#!/bin/bash
# EXPORT LIST
# dataset
# scale
# dtype
# dist
# file_in
# file_q
# file_gt
# lsh
# rr
# th
# L
# rot
RESULT_PATH=${RESULT_PREFIX}/result/FALCONN/$dataset/L${L}_rot${rot}

#set -x
date

mkdir -p $RESULT_PATH

echo "Running for the first ${scale} million points on ${dataset}"
param_basic="-n $((scale*1000000)) -type ${dtype}"
param_building="-dist ${dist} -in ${file_in} -lsh ${lsh}"
param_query="-q ${file_q} -g ${file_gt} -r ${rr} -th ${th} -l ${L}"
param_other="-rot ${rot}"
echo "./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${RESULT_PATH}/${scale}M.log 2>&1"
./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${RESULT_PATH}/${scale}M.log 2>&1

