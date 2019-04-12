## TO BUILD CONTAINER:
## Make sure you have downloaded the Linux version of OpenVINO: https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux
## Place the downloaded OpenVINO installer in the same directory as this Dockerfile.
## docker build -t unet_openvino --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg NO_PROXY=${NO_PROXY} --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
## TO RUN BUILT CONTAINER:
## 1. For Neural Compute Stick 2 - `docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix --privileged -v /dev:/dev -it unet_openvino`
## 2. For CPU - `docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix  -it unet_openvino`
FROM ubuntu:16.04

ARG OPENVINO_DIR=/opt/intel/openvino
ENV OPENVINO_PLUGIN_PATH ${OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64

ENV APP_DIR /app
ADD . ${APP_DIR}
WORKDIR ${APP_DIR}

ENV PATH /opt/conda/bin:$PATH

# Make sure programs are installed
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        build-essential \
        curl \
        cpio \
        cmake \
        git \
        g++ \
        libomp-dev \
        libtool \
        lsb-release \
        nano \
        pciutils \
        python3.5 \
        python3-pip \
        python3-setuptools \
        qt5-qmake qtcreator qt5-default \
        sudo \
        tar \
        udev \
        unzip \
        usbutils \
        wget \
        libgtk2.0-dev \
        libcanberra-gtk-module \
        libgflags-dev \
        vim \
        && apt-get clean all

# Install miniconda
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -y -n base -c defaults conda

# We need a python environment for OpenVINO
RUN conda create -y -n openvino python pip h5py numpy matplotlib tensorflow keras
RUN echo "conda activate openvino" >> ~/.bashrc

# Unzip the OpenVINO installer
RUN cd ${APP_DIR} && tar -xvzf l_openvino_toolkit*

# installing OpenVINO dependencies
RUN cd ${APP_DIR}/l_openvino_toolkit* && \
    ./install_openvino_dependencies.sh

# installing OpenVINO itself
RUN cd ${APP_DIR}/l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Build the samples so that we have libraries
RUN /bin/bash -c "${OPENVINO_DIR}/inference_engine/samples/build_samples.sh"

RUN echo "source ${OPENVINO_DIR}/bin/setupvars.sh" >> ~/.bashrc
RUN useradd -G video -ms /bin/bash user

# USB rules for Myriad
RUN cp ${APP_DIR}/97-myriad-usbboot.rules /etc/udev/rules.d/
RUN echo "udevadm control --reload-rules" >> ~/.bashrc
RUN echo "udevadm trigger" >> ~/.bashrc

# Build C++
RUN ${APP_DIR}/build_unet_demo_c++.sh

# Cleanup
RUN rm -rf ${APP_DIR}/l_openvino_toolkit*
RUN rm -f ${APP_DIR}/Dockerfile
RUN rm -f ${APP_DIR}/97-myriad-usbboot.rules
RUN rm -f ${APP_DIR}/README.md
RUN rm -f ${APP_DIR}/build_docker_container.sh
RUN rm -f ${APP_DIR}/create_validation_sample.py
RUN rm -rf ${APP_DIR}/create_openvino_model.sh

CMD ["/bin/bash"]
