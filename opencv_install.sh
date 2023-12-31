sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# # Create build directory
mkdir -p build && cd build
# # Configure
cmake  ../opencv-4.x
# # Build
cmake --build . -j 24