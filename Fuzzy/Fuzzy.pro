TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui


SOURCES += main.cpp \
    fuzzycmeans.cpp \
    fuzzypca.cpp

HEADERS += \
    fuzzycmeans.h \
    fuzzypca.h
