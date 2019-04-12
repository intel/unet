QT -= gui
QT += widgets

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


DESTDIR = ../build
OBJECTS_DIR =  ../build

SOURCES += ../src/main.cpp \
    ../src/brainunetopenvino.cpp

HEADERS +=  ../include/brainunetopenvino.h

INCLUDEPATH += "../src/cnpy/"   #to read the numpy arrays in c++


INCLUDEPATH += "${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/include/"
INCLUDEPATH += "${INTEL_OPENVINO_DIR}/opencv/include/"
LIBS += -L"${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/"
LIBS += -L"${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/omp/lib/"
LIBS += -L"${INTEL_OPENVINO_DIR}/inference_engine/lib/intel64/"
LIBS += -lcpu_extension_avx512 -lcnpy
LIBS += -L"${INTEL_OPENVINO_DIR}/inference_engine/external/omp/lib/"
LIBS += -liomp5
LIBS += -L"${INTEL_OPENVINO_DIR}/deployment_tools/external/mkltiny_lnx/lib/"
LIBS += -linference_engine -ldl
LIBS += -L"${INTEL_OPENVINO_DIR}/opencv/lib/"
LIBS +=-lopencv_highgui \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_videoio \
    -lopencv_imgcodecs \
    -lopencv_features2d\
    -lopencv_objdetect \
    -lopencv_video

INCLUDEPATH += "/usr/local/include/"
LIBS += -L"/usr/local/lib/"
LIBS += -lcnpy \
    -lz
