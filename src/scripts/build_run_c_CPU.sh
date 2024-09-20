export OMP_NUM_THREADS=4  # Use 4 threads

cd your_project/src/c
mkdir build
cd build
cmake .. -DUSE_CUDA=OFF
make
