FROM  nvcr.io/nvidia/tritonserver:23.08-py3
LABEL maintainer="Bobo" description="triton serving including models"


# Copy all models to docker
COPY ./models /models
COPY ./opencv_install.sh /opt/tritonserver/

# tritonserver 的所有backbends 都在 /opt/tritonserver/backbends 下，需要新建一个目录
RUN mkdir /opt/tritonserver/backends/image_process

# 将生成的.so 文件复制到backbend 目录下
# COPY ./libtriton_image_process.so /opt/tritonserver/backends/image_process/

# install sudo
RUN apt-get update && apt-get -y install sudo

# install rapidjson
RUN sudo apt-get install -y rapidjson-dev

# install opencv
RUN bash opencv_install.sh

ENV LD_LIBRARY_PATH="/opt/tritonserver/build/lib/:${LD_LIBRARY_PATH}"
ENV OpenCV_DIR="/opt/tritonserver/build/"


# RUN echo -e '#!/bin/bash \n\n\
# tritonserver --model-repository=/models \
# "$@"' > /usr/bin/triton_serving_entrypoint.sh \
# && chmod +x /usr/bin/triton_serving_entrypoint.sh

# ENTRYPOINT ["/usr/bin/triton_serving_entrypoint.sh"]
