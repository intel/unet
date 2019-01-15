#cd openvino_saved_model
#python inference_openvino.py -l ${INTEL_CVSDK_DIR}/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so --plot --stats -d MYRIAD
python inference_openvino.py -l ${INTEL_CVSDK_DIR}/inference_engine/lib/centos_7.4/intel64/libcpu_extension_avx2.so --plot --stats -d MYRIAD


