TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    Bbox.cpp \
    intergral_blur.cpp \
    surf_match.cpp \
    edcirlces.cpp \
    tag_generator.cpp \
    thread_demo.cpp

INCLUDEPATH += /usr/local/include/opencv \
               /usr/local/include/opencv2 \
               /usr/local/include

LIBS += /usr/local/lib/libopencv_video.so  \
  /usr/local/lib/libopencv_objdetect.so \
  /usr/local/lib/libopencv_ml.so  \
  /usr/local/lib/libopencv_core.so \
  /usr/local/lib/libopencv_features2d.so  \
  /usr/local/lib/libopencv_imgproc.so \
  /usr/local/lib/libopencv_highgui.so \
  /usr/local/lib/libopencv_flann.so   \
  /usr/local/lib/libopencv_calib3d.so \
  /usr/local/lib/libopencv_imgcodecs.so

DISTFILES += \
    CMakeLists.txt
