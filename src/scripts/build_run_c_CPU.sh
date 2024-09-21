export OMP_NUM_THREADS=4  # Use 4 threads

cd ../c
mkdir build
cd build
cmake .. -DUSE_CUDA=OFF
make
