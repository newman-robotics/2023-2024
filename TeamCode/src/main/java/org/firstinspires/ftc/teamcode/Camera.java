package org.firstinspires.ftc.teamcode;

import android.graphics.Bitmap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import androidx.annotation.NonNull;

import org.firstinspires.ftc.robotcore.external.function.Consumer;
import org.firstinspires.ftc.robotcore.external.function.Continuation;
import org.firstinspires.ftc.robotcore.external.function.ContinuationResult;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureRequest;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureSequenceId;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureSession;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraFrame;
import org.firstinspires.ftc.robotcore.external.stream.CameraStreamClient;
import org.firstinspires.ftc.robotcore.external.stream.CameraStreamSource;
import org.slf4j.LoggerFactory;

public class Camera {
    public static class DefaultCallback implements CameraCaptureSession.CaptureCallback, CameraCaptureSession.StateCallback, CameraCaptureSession.StatusCallback {
        public native void onConfigured(@NonNull CameraCaptureSession session);
        public void onClosed(@NonNull CameraCaptureSession session) {}
        public native void onNewFrame(@NonNull CameraCaptureSession session, @NonNull CameraCaptureRequest request, @NonNull CameraFrame frame);
        public void onCaptureSequenceCompleted(@NonNull CameraCaptureSession session, CameraCaptureSequenceId id, long somethingOrOther) {}
    }

    public static class FrameSource implements CameraStreamSource {
        //I don't want to handle the Continuation stuff in C++
        @Override
        public void getFrameBitmap(Continuation<? extends Consumer<Bitmap>> continuation) {
            continuation.dispatch((ContinuationResult<Consumer<Bitmap>>)(Consumer<Bitmap> frameAcceptingFunction) -> {
                Bitmap toSend = Bitmap.createBitmap(this.getFrame(), 1280, 720, Bitmap.Config.RGB_565);
                frameAcceptingFunction.accept(toSend);
                LoggerFactory.getLogger("camera-server").info("Recieved frame " + toSend.toString() + " from C++, no segfault...");
            });
        }

        public native int[] getFrame();
    }

    public static class FrameListener implements CameraStreamClient.Listener {

        @Override
        public void onStreamAvailableChange(boolean available) {
            LoggerFactory.getLogger("camera-client").info("Camera stream is available: " + available);
        }

        @Override
        public void onFrameBitmap(Bitmap frameBitmap) {
            LoggerFactory.getLogger("camera-client").info("Received frame from server, no segfault...");
        }
    }
}
