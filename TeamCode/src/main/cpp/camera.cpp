#include "include/jni.h"
#include "libcardinal.hpp"
#include "threadutils.hpp"
#include "include/opencv2/core.hpp"
#include "include/opencv2/imgproc.hpp"

#include <numeric>
#include <utility>

#include <opencv2/dnn.hpp>

JavaVM * vm;
JNIEnv * frameHolderEnv;

std::vector<char> lastFrame;
SuppliedObjectHolder<cv::Mat*> frameHolder;
std::unique_lock<std::mutex> frameLock;
jobject logger;
bool isDead = false;

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Camera_runOpMode(JNIEnv * env, jobject thiz) {
    //Because we are multithreading, we need the Java VM so that our other threads can get their environments.
    env->GetJavaVM(&vm);
    libcardinal::setenv(env);

    //We proceed to create the logger. (I can't get telemetry to work, so I use loggers.)
    jvalue string = {.l=env->NewStringUTF("camera")};
    logger = env->NewGlobalRef(libcardinal::call_static(env->FindClass("org/slf4j/LoggerFactory"), "getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", &string).l);

    //Now we get the camera manager, from which we will get our camera.
    jobject classFactory = libcardinal::call_static(env->FindClass("org/firstinspires/ftc/robotcore/external/ClassFactory"), "getInstance", "()Lorg/firstinspires/ftc/robotcore/external/ClassFactory;", nullptr).l;
    jobject cameraManager = libcardinal::call_instance(classFactory, "getCameraManager", "()Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraManager;", nullptr).l;

    //Now we get the name of the camera.
    jvalue seconds = libcardinal::get_static_field(env->FindClass("java/util/concurrent/TimeUnit"), "SECONDS", "Ljava/util/concurrent/TimeUnit;");
    jvalue targs[2] = {{.j=600}, seconds};
    jobject deadline = libcardinal::new_instance("org/firstinspires/ftc/robotcore/internal/system/Deadline", "(JLjava/util/concurrent/TimeUnit;)V", targs);
    jobject cameraName = libcardinal::get_device_from_hardware_map(thiz, "webcam", "org/firstinspires/ftc/robotcore/external/hardware/camera/CameraName");

    //Now, for the fun part: we actually create the camera.
    jvalue args[3] = {{.l=deadline}, {.l=cameraName}, {.l=nullptr}};
    jobject camera2 = libcardinal::call_instance(cameraManager, "requestPermissionAndOpenCamera", "(Lorg/firstinspires/ftc/robotcore/internal/system/Deadline;Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraName;Lorg/firstinspires/ftc/robotcore/external/function/Continuation;)Lorg/firstinspires/ftc/robotcore/external/hardware/camera/Camera;", args).l;

    //Now we create the request to the camera to actually get frames.
    targs[0] = {.i=1280}; //width STATIC
    targs[1] = {.i=720}; //height STATIC
    jobject size = libcardinal::new_instance("org/firstinspires/ftc/robotcore/external/android/util/Size", "(II)V", targs);
    args[0] = {.i=20}; //android format DO NOT CHANGE
    args[1] = {.l=size}; //size; see above
    args[2] = {.i=30}; //fps
    jobject request = libcardinal::call_instance(camera2, "createCaptureRequest", "(ILorg/firstinspires/ftc/robotcore/external/android/util/Size;I)Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraCaptureRequest;", args).l;

    //Now we do some strange threading stuff to get our camera to start sending us frames.
    jvalue callback = {.l=libcardinal::new_instance("org/firstinspires/ftc/teamcode/Camera$DefaultCallback","()V", nullptr)};
    jvalue continuation = libcardinal::call_static(env->FindClass("org/firstinspires/ftc/robotcore/external/function/Continuation"), "createTrivial", "(Ljava/lang/Object;)Lorg/firstinspires/ftc/robotcore/external/function/Continuation;", &callback);
    jobject session = libcardinal::call_instance(camera2, "createCaptureSession", "(Lorg/firstinspires/ftc/robotcore/external/function/Continuation;)Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraCaptureSession;", &continuation).l;
    jvalue cargs[3] = {{.l=request}, continuation, continuation};
    libcardinal::call_instance(session, "startCapture", "(Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraCaptureRequest;Lorg/firstinspires/ftc/robotcore/external/function/Continuation;Lorg/firstinspires/ftc/robotcore/external/function/Continuation;)Lorg/firstinspires/ftc/robotcore/external/hardware/camera/CameraCaptureSequenceId;", cargs);

    //This is one of my utility classes, which takes a lambda function and calls it whenever the camera gets a new frame.
    frameHolder.async_capture([](cv::Mat * arg){
        //If the frame is not good, we don't use it. That may lead to issues down the line, with us processing the same frame twice, but hey, we didn't get a frame...
        if (arg->empty() or not arg->isContinuous()) {
            libcardinal::altenv_log(frameHolderEnv, logger, "Frame is empty or non-continuous. Skipping.");
            return;
        }
        //We set up the frame.
        lastFrame.assign(arg->ptr(), arg->ptr() + 921600);
        libcardinal::altenv_log(frameHolderEnv, logger, "Ready...");
    }, [](){
        //Create an environment when the new thread loads. Hasn't crashed yet...
        vm->AttachCurrentThread((void**)(&frameHolderEnv), nullptr);
    });

    //Now we are finally ready to send our frames to the Driver Station.
    jobject cameraStreamServer = libcardinal::call_static(env->FindClass("org/firstinspires/ftc/robotcore/external/stream/CameraStreamServer"), "getInstance", "()Lorg/firstinspires/ftc/robotcore/external/stream/CameraStreamServer;", nullptr).l;
    jvalue source = {.l=libcardinal::alloc_instance("org/firstinspires/ftc/teamcode/Camera$FrameSource")};
    libcardinal::call_void_instance(cameraStreamServer, "setSource", "(Lorg/firstinspires/ftc/robotcore/external/stream/CameraStreamSource;)V", &source);

    //Once the user presses the start button, we stop all of our processing and terminate.
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    isDead = true;
    env->DeleteGlobalRef(logger);
}

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Camera_00024DefaultCallback_onConfigured(JNIEnv *env, jobject thiz, jobject cameraCaptureSession) {
    frameLock = frameHolder.lock();
}

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Camera_00024DefaultCallback_onNewFrame(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jobject session,
                                                                           jobject request,
                                                                           jobject frame) {
    if (not isDead) {
        jbyteArray data = (jbyteArray)libcardinal::altenv_call_instance(env, frame, "getImageData", "()[B", nullptr).l;
        auto * frame2 = (unsigned char*)env->GetByteArrayElements(data, nullptr);
        int size[3] = {1280, 720};
        cv::Mat img(2, size, CV_8UC2, frame2);
        cv::Mat out;
        cv::cvtColor(img, out, cv::COLOR_YUV2GRAY_YUYV);
        frameHolder.set(&out, frameLock);
    }
}

extern "C" JNIEXPORT jintArray JNICALL Java_org_firstinspires_ftc_teamcode_Camera_00024FrameSource_getFrame(JNIEnv *env, jobject thiz) {

    jintArray out = env->NewIntArray(921600);
    std::vector<int> lastFrameAsInt(lastFrame.begin(), lastFrame.end());
    env->SetIntArrayRegion(out, 0, 921600, (jint *) (&(lastFrameAsInt.front())));
    return out;
}