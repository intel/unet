# Docker 

There are now official Intel Docker images. Please see the full instructions at https://hub.docker.com/r/openvino/ubuntu18_dev_no_samples

## Intel Neural Compute Stick 2 (NCS2)
If you need Myriad* (NCS2) accelerator only, run image with the following command:
```
docker run -it --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --rm openvino/ubuntu18_dev_no_samples:latest
```

## For CPU only, run this command:
```
docker run -it --rm openvino/ubuntu18_dev_no_samples:latest
```
