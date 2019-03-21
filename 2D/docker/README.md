# Docker 

To build the Docker container:
1. Make sure that you have trained the model and converted it to OpenVINO IR format. The instructions for that are [here](https://github.com/IntelAI/unet/blob/master/2D/openvino/README.md).
2. You'll need to [download Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux). You'll want the Linux version of the installer since the Docker container will use Linux as its operating system. Don't worry, the final Docker container can run on any operating system that supports Docker (e.g. Linux, Windows, Mac)
3. The OpenVINO installer should be named something like `l_openvino_toolkit_p_2018.5.445.tgz`. Move it to the `unet/single-node/docker` subdirectory.
4. Run the [build bash](https://github.com/IntelAI/unet/blob/master/2D/docker/build_docker_container.sh) script. ```./build_docker_container.sh```

A build log should begin printing. The build time will heavily depend on your local hardware. If it finishes succesfully, you should see something like this:

![docker build](https://github.com/IntelAI/unet/blob/master/2D/images/docker_build.png)

To run the Docker container:
1. If the docker container has been successfully built, you can run it with the command: 

```docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix  -it unet_openvino```

2. If you have an [Intel Neural Compute Stick (NCS)](https://software.intel.com/en-us/neural-compute-stick), plug it into the USB port and run the Docker this way: 

```docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix --privileged -v /dev:/dev -it unet_openvino```

Once the Docker starts you'll be in a new shell. To run the OpenVINO inference script type:

```./run_unet_demo_cpu.sh``` 

or (for NCS):

```./run_unet_demo_ncs.sh``` 

The script will run the OpenVINO model on a few sample MRIs from the [Medical Decathlon dataset](http://medicaldecathlon.com/) ([CC BY-SA4 license](https://creativecommons.org/licenses/by-sa/4.0/)). (These samples were generated during the OpenVINO [conversion step #4](https://github.com/IntelAI/unet/blob/master/2D/openvino/README.md)). It will show you the Dice scores and plot/save some PNG images into a subdirectory.

![docker run](https://github.com/IntelAI/unet/blob/master/2D/images/docker_run.png)

 
