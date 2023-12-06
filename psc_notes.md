# Research Steps

## Environment Preparation

1. Install packages from conda

```shell
conda install pytorch torchvision  pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy numba tqdm
```

2. Install packages from pip

Since PyTorch-GPU and OpenCV cannot be installed simultaneously using conda, the reason is unknown. We need to install OpenCV via pip.

```shell
pip install opencv-python
```

3. Build and install Open3D

Open3D's release is not so active. Latest release does not work well with newer Python and GCC-13. We need to compile by our own.

1. Clone the modified repository

```shell
git clone git@github.com:psc1453/Open3D.git
```

2. Install build dependencies

Refer to `util/install_deps_ubuntu.sh`. If you are using Debian Ubuntu or other Debian-Based distribution which uses APT as package managing tool. You can run the script directly. Otherwise, check the content of the file and install dependencies in the `deps` list.

3. Install older GCC

This needs to be done by your self. GCC-13 is not supported. You can install GCC-12.

4. Modify CMakeList

You can find the code below on the top of `CMakeList.txt`

```cmake
set(CMAKE_C_COMPILER "gcc-12")
set(CMAKE_CXX_COMPILER "g++-12")
```

Modify the content to the path of older GCC which you installed.

5. Compile and install

Activate the virtual environment you want to use.

```shell
mkdir build && cd build
cmake ..
make -j$(nproc)
make install-pip-package
```

## Compile CUDA Operators

In PointPillars project, two operators are written in CUDA, they are IOU3D and Voxelization. Their source code is located in `ops` directory and you need to compile them first.

```shell
cd ops
python setup.py develop
```

You may encounter some problems like `/lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, this bug is caused by the library conda use is lower than that in the system. You can use the following command to link system library to your conda environment.

```shell
ln -sf /usr/lib/libstdc++.so.6 /home/psc/anaconda3/envs/[NAME_OF_YOUR_ENVIRONMENT]/lib/libstdc++.so.6
```

## Inference

```shell
python test.py --ckpt pretrained/epoch_160.pth --pc_path kitti/training/velodyne/000100.bin --calib_path kitti/training/calib/000100.txt  --gt_path kitti/training/label_2/000100.txt --img_path kitti/training/image_2/000100.png
```