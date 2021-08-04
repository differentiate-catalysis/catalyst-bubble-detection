#!/bin/bash

# Pick the version of OpenCV to download
OPENCV_VERSION='4.5.3'

# Dependencies
echo 'Installing dependencies from APT...'
sudo apt -y update >> /dev/null 2>&1
sudo apt install -y build-essential cmake qt5-default libvtk6-dev qtbase5-dev \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev \
    libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev \
    libtbb-dev libeigen3-dev \
    python-dev python-tk pylint python-numpy python3-dev python3-tk pylint3 python3-numpy flake8 \
    ant default-jdk >> /dev/null 2>&1

# Check for environment things
echo 'Are you using conda? (Y/n)'
read using_conda
using_venv='n'
if [ $using_conda = 'Y' ] || [ $using_conda = 'y' ]; then
    echo 'Please enter your conda environment:'
    read conda_env
    eval "$(conda shell.bash hook)"
    conda activate $conda_env
    echo 'Installing numpy in environment' $conda_env
    conda install numpy -y >> /dev/null
else
    echo 'Are you using virtualenv? (Y/n)'
    read using_venv

    if [ $using_venv = 'Y' ] || [ $using_venv = 'y' ]; then
        echo 'Please enter the directory of your virtualenv'
        read venv
        source_path="$venv/bin/activate"
        unfolded_source_path=$(python3 -c "import os.path;print(os.path.expanduser(\"$source_path\"))")
        source $unfolded_source_path
        echo 'Installing numpy in environment'
        python3 -m pip install numpy >> /dev/null
    fi
fi

# Check for existing OpenCV build
if [ -d 'OpenCV' ]; then
    echo 'Found an existing OpenCV directory! Re-use this download? (Y/n)'
    read reuse_download
    if [ $reuse_download = 'Y' ] || [ $reuse_download = 'y' ]; then
        cd OpenCV
    else
        rm -rf OpenCV
        # Download
        echo 'Downloading OpenCV' $OPENCV_VERSION 'source'
        wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip >> /dev/null
        unzip ${OPENCV_VERSION}.zip >> /dev/null
        rm ${OPENCV_VERSION}.zip 
        mv opencv-${OPENCV_VERSION} OpenCV

        echo 'Downloading OpenCV contributor source'
        wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip >> /dev/null
        unzip ${OPENCV_VERSION}.zip >> /dev/null
        rm ${OPENCV_VERSION}.zip 
        mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
        mv opencv_contrib OpenCV
        cd OpenCV
    fi
else
    # Download
    echo 'Downloading OpenCV' $OPENCV_VERSION 'source'
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip >> /dev/null
    unzip ${OPENCV_VERSION}.zip >> /dev/null
    rm ${OPENCV_VERSION}.zip 
    mv opencv-${OPENCV_VERSION} OpenCV

    echo 'Downloading OpenCV contributor source'
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip >> /dev/null
    unzip ${OPENCV_VERSION}.zip >> /dev/null
    rm ${OPENCV_VERSION}.zip 
    mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
    mv opencv_contrib OpenCV
    cd OpenCV
fi



# Build
if [ -d "build" ]; then
    echo 'Found an existing build directory! Re-use this build? (Y/n)'
    read reuse_build
    if [ $reuse_build = 'Y' ] || [ $reuse_build = 'y' ]; then
        cd build
        rm CMakeCache.txt
        rm -rf CMakeFiles
        rm cmake_install.cmake
        rm CMakeVars.txt
    else
        rm -rf build
        mkdir build
        cd build
    fi
else
    mkdir build && cd build
fi

if [ $using_conda = 'Y' ] || [ $using_conda = 'y' ]; then
    python_exec=`which python3`
    include_dir=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
    library=`python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`
    packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
    stdlib=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])'`
    export CPLUS_INCLUDE_PATH=$stdlib
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
          -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF \
          -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON -DWITH_CUFFT=ON \
          -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
          -DPYTHON3_EXECUTABLE=$python_exec -DPYTHON3_INCLUDE_DIR=$include_dir -DPYTHON3_LIBRARY=$library -DPYTHON3_PACKAGES_PATH=$packages \
          -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules .. #>> /dev/null 2>&1
elif [ $using_venv = 'Y' ] || [ $using_venv = 'y' ]; then
    python_exec=`which python3`
    include_dir=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
    library=`python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`
    packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
    stdlib=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["platstdlib"])'`
    export CPLUS_INCLUDE_PATH=$stdlib
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
          -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF \
          -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON -DWITH_CUFFT=ON \
          -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
          -DPYTHON3_EXECUTABLE=$python_exec -DPYTHON3_PACKAGES_PATH=$packages \
          -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules .. #>> /dev/null 2>&1
else
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
          -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF \
          -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON -DWITH_CUFFT=ON \
          -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
          -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules .. #>> /dev/null 2>&1
fi
cores=$nproc
make -j $cores
sudo make install
sudo ldconfig

