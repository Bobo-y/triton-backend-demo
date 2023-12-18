1. build custom backend

```shell
sudo docker build -t serving-triton .
sudo docker run --gpus all -it -v /home/:/home/ -p 8000:8000 -p 8001:8001 -p 8002:8002 serving-triton  /bin/bash


# cd image_process dir
mkdir build & cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install -j8

# for debug cuda kernel
mv image_process.cc image_process.cc.bak
mv image_process_cuda.cc  image_process.cc
mkdir build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install -j8
```

2. start custom backend server
# in docker
```shell
# cd /opt/tritonserver
cp /home/XXX/backend_demo/image_process/build/libimageprocess.so /opt/tritonserver/backends/image_process/
mv /opt/tritonserver/backends/image_process/libimageprocess.so /opt/tritonserver/backends/image_process/libtriton_image_process.so
./bin/tritonserver --model-repository=/models
```

3. client 
outside docker

```python
# please use absolute path of test imgs
python client.py
```
