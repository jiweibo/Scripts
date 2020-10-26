mkdir -p build
cd build
rm -rf *

LIB_DIR=$1
DEMO_NAME=$2
WITH_MKL=ON
WITH_GPU=OFF

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \

make -j
