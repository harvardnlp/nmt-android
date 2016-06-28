package com.torch;

import android.util.Log;
import android.content.res.AssetManager;
import android.content.pm.ApplicationInfo;
import android.content.Context;
import android.os.Bundle;
import android.app.Activity;
import com.torch.torchdemo.R;



import android.content.Context;
import android.content.res.AssetManager;
import android.content.pm.ApplicationInfo;

public class Torch {

    private final String TAG = "LuaManager";
    private static Torch mInstance;

    private long mTorchState;
    static {
        System.loadLibrary("torchdemo");
    }

    //private native int getTopResult(long stateLocation, int width, int height,
    //                                byte[] bitmapRGBData);

    //private native float testTorchData(long stateLocation);

    private native long initTorch(AssetManager manager, String libdir);

    private native void destroyTorch(long stateLocation);

    private native String translateSentence(long stateLocation, String sent);
    //private native long translateSentence(long stateLocation, String sent);

    public static Torch getLuaManager(Context context) {
        if (mInstance == null)
            mInstance = new Torch(context);
        return mInstance;
    }

    //public int getTopRecognitionResult(int width, int height,
    //                                   byte[] bitmapRGBData) {
    //    return getTopResult(mTorchState, width, height, bitmapRGBData);
    //}

    private Torch(Context context) {
        ApplicationInfo info = context.getApplicationInfo();
        mTorchState = initTorch(context.getAssets(), info.nativeLibraryDir);
    }

    @Override
    protected void finalize() throws Throwable {
        destroyTorch(mTorchState);
        super.finalize();
    }

    public String translateSentence(String sent) {
        return translateSentence(mTorchState, sent);
        //return "bye bye";
    }
}

