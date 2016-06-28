/*
 * Copyright (C) 2013 Soumith Chintala
 *
 */
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include "torchandroid.h"
#include <assert.h>
#include <android/log.h>

extern "C" {

JNIEXPORT long JNICALL
Java_com_torch_Torch_initTorch(JNIEnv *env, jobject thiz, jobject assetManager,
                                                         jstring nativeLibraryDir_)
{
        // get native asset manager. This allows access to files stored in the assets folder
        AAssetManager* manager = AAssetManager_fromJava(env, assetManager);
        assert( NULL != manager);

        const char *nativeLibraryDir = env->GetStringUTFChars(nativeLibraryDir_, 0);

        lua_State *L = NULL;
        L = inittorch(manager, nativeLibraryDir);// create a lua_State

        // load and run file
        char file[] = "main.lua";
        int ret;
        long size = android_asset_get_size(file);
        if (size != -1) {
                char *filebytes = android_asset_get_bytes(file);
                ret = luaL_dobuffer(L, filebytes, size, "main");
                if (ret == 1) {
                        D("Torch Error doing resource: %s\n", file);
                        D(lua_tostring(L,-1));
                } else {
                        D("Torch script ran succesfully.");
                }
        }

        lua_getglobal(L, "load_model");
        //lua_pushstring(L, sent);
        //luaT_pushudata(L, testTensor, "torch.DoubleTensor");
        if (lua_pcall(L, 0, 1, 0) != 0) {
                         __android_log_print(ANDROID_LOG_INFO, "Torchandroid",
                                         "Error running function: %s", lua_tostring(L, -1));
        }
        return (long) L;
}
JNIEXPORT void JNICALL
Java_com_torch_Torch_destroyTorch(JNIEnv *env, jobject thiz,jlong torchStateLocation)
{
        lua_State *L = (lua_State*) torchStateLocation;
        lua_close(L); //Close lua state.
}

JNIEXPORT jstring JNICALL
Java_com_torch_Torch_translateSentence(JNIEnv *env, jobject thiz, jlong torchStateLocation, jstring path_in) {

        lua_State *L = (lua_State*) torchStateLocation;
        char *sent;
        sent = const_cast<char*> ( env->GetStringUTFChars(path_in , NULL ) );
        //sent = env->GetStringUTFChars(env, path_in, NULL ) ;
        jstring result;
        //int size = width;
        //THDoubleTensor *testTensor = THDoubleTensor_newWithSize1d(3 * size); //Initialize 1D tensor with size * 3 (R,G,B).
        //jdouble *testTensorData = THDoubleTensor_data(testTensor);
        //jbyte *inputData;//Initialize tensor to store java byte data from bitmap.
        //inputData = (env)->GetByteArrayElements(bitmapRGBData,0);//Get pointer to java byte array region

        //for (int i = 0; i < size; i++) {
                //convert Byte value to int by &0xff and to save it as double
                //save R G B seperatly as cifar-10 data set
        //        testTensorData[i] = inputData[i * 4] & 0xFF;//R
        //        testTensorData[size + i] = inputData[i * 4 + 1] & 0xFF;//G
        //        testTensorData[size * 2 + i] = inputData[i * 4 + 2] & 0xFF;//B
        //}

        lua_getglobal(L, "main");
        lua_pushstring(L, sent);
        //luaT_pushudata(L, testTensor, "torch.DoubleTensor");
        const char * trans;
        if (lua_pcall(L, 1, 1, 0) == 0) {
                //Call function. Print error if call not successful

        //} else {
                trans = lua_tostring(L, -1);
                //result = (float) lua_tonumber(L,-1);
         } else  {
                         __android_log_print(ANDROID_LOG_INFO, "Torchandroid",
                                         "Error running function: %s", lua_tostring(L, -1));
        //        lua_pop(L,1);
         //       __android_log_print(ANDROID_LOG_INFO, "Torchandroid", "result : %d",result);
        }
        //env->ReleaseByteArrayElements(bitmapRGBData, inputData, 0); //Destroy pointer to location in C. Only need java now
        result = env->NewStringUTF(trans);
        return result;
        //return result;
}



}
