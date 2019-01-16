## TO BUILD CONTAINER:
## docker build -t unet_openvino --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg NO_PROXY=${NO_PROXY} --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
## TO RUN BUILT CONTAINER:
## 1. For Neural Compute Stick 2 - `docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix --privileged -v /dev:/dev -it unet_openvino`
## 2. For CPU - `docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix  -it unet_openvino`
FROM ubuntu:16.04

ADD . /app
WORKDIR /app

ARG INSTALL_DIR=/opt/intel/computer_vision_sdk

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update

# Make sure programs are installed
RUN apt-get install -y --no-install-recommends \
        autoconf \
        build-essential \
        curl \
        cpio \
        cmake \
        git \
        g++ \
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

RUN conda update -n base -c defaults conda

# We need a python 3.5 environment for OpenVINO
RUN conda create -n openvino python=3.5 pip h5py numpy matplotlib
RUN echo "conda activate openvino" >> ~/.bashrc

# Unzip the OpenVINO installer
RUN cd /app && tar -xvzf l_openvino_toolkit*

# installing OpenVINO dependencies
RUN cd /app/l_openvino_toolkit* && \
    ./install_cv_sdk_dependencies.sh

# installing OpenVINO itself
RUN cd /app/l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Setup the OpenVINO environment
RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"

# Build the samples so that we have libraries
RUN /bin/bash -c "$INSTALL_DIR/inference_engine/samples/build_samples.sh"

RUN echo "source ${INSTALL_DIR}/bin/setupvars.sh" >> ~/.bashrc
RUN useradd -G video -ms /bin/bash user

# USB rules for Myriad
RUN cp /app/97-myriad-usbboot.rules /etc/udev/rules.d/
RUN echo "udevadm control --reload-rules" >> ~/.bashrc
RUN echo "udevadm trigger" >> ~/.bashrc

RUN chmod a+x /app/run_unet_demo_cpu.sh
RUN chmod a+x /app/run_unet_demo_movidius.sh

CMD ["/bin/bash"]
