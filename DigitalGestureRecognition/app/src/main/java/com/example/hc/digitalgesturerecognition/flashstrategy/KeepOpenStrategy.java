package com.example.hc.digitalgesturerecognition.flashstrategy;

import android.annotation.SuppressLint;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.os.Handler;

/**
 * Created by ice on 2017/12/3.
 */

public class KeepOpenStrategy implements FlashStrategy {
    @SuppressLint("NewApi")
    @Override
    public void setCaptureRequest(CaptureRequest.Builder requestBuilder, CameraCaptureSession cameraCaptureSession, Handler handler) {
        requestBuilder.set(CaptureRequest.CONTROL_AE_MODE,CaptureRequest.CONTROL_AE_MODE_ON);
        requestBuilder.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH);
        try {
            cameraCaptureSession.setRepeatingRequest(requestBuilder.build(),null,handler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
}
