TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lpthread


SOURCES += main.cpp \
    fuzzycmeans.cpp \
    fuzzypca.cpp \
    sil.cpp \
    silt.cpp \
    distancematrix.cpp

HEADERS += \
    fuzzycmeans.h \
    fuzzypca.h \
    sil.h \
    silt.h \
    distancematrix.h
