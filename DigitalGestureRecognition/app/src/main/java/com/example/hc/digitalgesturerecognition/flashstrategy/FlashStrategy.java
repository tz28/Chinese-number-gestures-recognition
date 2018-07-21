package com.example.hc.digitalgesturerecognition.flashstrategy;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.os.Handler;


/**
 * Created by ice on 2017/12/3.
 */

public interface FlashStrategy {

    void setCaptureRequest(CaptureRequest.Builder requestBuilder, CameraCaptureSession cameraCaptureSession,
                           Handler handler);

}
