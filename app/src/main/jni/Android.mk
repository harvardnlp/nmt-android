LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)

LOCAL_MODULE := torchdemo

LOCAL_C_INCLUDES += /home/srush/lib/torch-android/install/include/

LOCAL_SRC_FILES := torchdemo.cpp

LOCAL_LDLIBS := -L /home/srush/lib/torch-android/install/lib -L /home/srush/lib/torch-android/install/libs/armeabi-v7a  -lnnx -limage  -lTHNN -ltorch  -lTH -lluaT -lluajit -ltorchandroid -llog -landroid

include $(BUILD_SHARED_LIBRARY)
