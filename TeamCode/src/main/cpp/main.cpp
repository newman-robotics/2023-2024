#include <memory>
#include <jni.h>
#include "libcardinal.hpp"
#include <apriltags/TagDetector.h>
#include <android/log.h>

enum MotorLocation {
    FRONT_LEFT_MOTOR,
    FRONT_RIGHT_MOTOR,
    BACK_LEFT_MOTOR,
    BACK_RIGHT_MOTOR,
};

//compilation params

//the name of the logger
#define  LOGGER_NAME "El J.U.L.I.O. L.I."

//whether to use driver oriented driving
#define  USE_DRIVER_ORIENTED false

//which one of these two uses of the right joystick should be used
//do NOT EVER enable both of them, or weird things will happen
//however, they can be both disabled
#define  USE_SWING false
#define  USE_ARM true

//certain other things
//that do things
#define  USE_MOTOR_CALIBRATION false
#define  USE_CLAW false

//which motor is the anchor, i.e. which motor should every other
//motor's ticks be pinned to
#define  ANCHOR FRONT_LEFT_MOTOR

//the speed multiplier of the motors
//0.0 is none, 1.0 is full
#define  SPEED 0.9
#define  ARM_SPEED 1.0

//how we implement print(...)
#define  print(...) __android_log_print(ANDROID_LOG_INFO,LOGGER_NAME,__VA_ARGS__)

static jvalue speed = {.d=ARM_SPEED};
static jvalue zero = {.d=0.0};
static jvalue neg_speed = {.d=-ARM_SPEED};

double toRadians(double degrees) {
    return (degrees * M_PI) / 180.0;
}

double toDegrees(double radians) {
    return (radians * 180.0) / M_PI;
}

template <typename T> struct OneValPerMotor {
    T fl;
    T fr;
    T bl;
    T br;
};

const OneValPerMotor<double> DEFAULT_VALUES = {.fl=1.0,.fr=1.0,.bl=1.0,.br=1.0};

struct Drivetrain {
    JNIEnv * env;
    jobject opMode;
    jobject frontLeft;
    jobject frontRight;
    jobject backLeft;
    jobject backRight;
    jobject imu;
    jobject gamepad;
    bool isDead;
#if USE_MOTOR_CALIBRATION
    OneValPerMotor<double> multipliers = DEFAULT_VALUES;
#endif

    Drivetrain(JNIEnv * env, jobject opMode) : isDead(false), env(env), opMode(env->NewGlobalRef(opMode)) {
        print("Setting up drivetrain...");
        this->frontLeft = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivefl", "com/qualcomm/robotcore/hardware/DcMotor"));
        print("Set up front left motor...");
        this->frontRight = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivefr", "com/qualcomm/robotcore/hardware/DcMotor"));
        print("Set up front right motor...");
        this->backLeft = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivebl", "com/qualcomm/robotcore/hardware/DcMotor"));
        print("Set up back left motor...");
        this->backRight = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivebr", "com/qualcomm/robotcore/hardware/DcMotor"));
        print("Set up back right motor...");
        this->imu = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "imu", "com/qualcomm/robotcore/hardware/IMU"));
        print("Set up IMU...");
        this->gamepad = env->NewGlobalRef(libcardinal::altenv_get_field(env, opMode, "gamepad1", "Lcom/qualcomm/robotcore/hardware/Gamepad;").l);
        print("Set up gamepad...");
        jvalue reversed = libcardinal::altenv_get_static_field(env, env->FindClass("com/qualcomm/robotcore/hardware/DcMotorSimple$Direction"), "REVERSE", "Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;");
        //libcardinal::call_void_instance(this->frontLeft, "setDirection", "(Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;)V", &reversed);
        libcardinal::call_void_instance(this->frontRight, "setDirection", "(Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;)V", &reversed);
        libcardinal::call_void_instance(this->imu, "resetYaw", "()V", nullptr);
    }
    ~Drivetrain() {
        print("Destructing drivetrain...");
        this->env->DeleteGlobalRef(this->frontLeft);
        this->env->DeleteGlobalRef(this->frontRight);
        this->env->DeleteGlobalRef(this->backLeft);
        this->env->DeleteGlobalRef(this->backRight);
        this->env->DeleteGlobalRef(this->imu);
        this->env->DeleteGlobalRef(this->gamepad);
    }

    void kill() {
#if USE_SWING
        this->update(0, 0, 0, 0, false);
#else
        this->update(0, 0, 0);
#endif
        this->isDead = true;
    }

#if USE_MOTOR_CALIBRATION
    [[nodiscard]] OneValPerMotor<int> getEncoderTicks() const {
        if (this->isDead) return {.fl=0,.fr=0,.bl=0,.br=0};
        int frontLeftTicks = abs(libcardinal::altenv_call_instance(this->env, this->frontLeft, "getCurrentPosition", "()I", nullptr).i);
        int frontRightTicks = abs(libcardinal::altenv_call_instance(this->env, this->frontRight, "getCurrentPosition", "()I", nullptr).i);
        int backLeftTicks = abs(libcardinal::altenv_call_instance(this->env, this->backLeft, "getCurrentPosition", "()I", nullptr).i);
        int backRightTicks = abs(libcardinal::altenv_call_instance(this->env, this->backRight, "getCurrentPosition", "()I", nullptr).i);
        return {.fl=frontLeftTicks, .fr=frontRightTicks, .bl=backLeftTicks, .br=backRightTicks};
    }

    void resetEncoders() const {
        if (this->isDead) return;
        jclass runmode_class = this->env->FindClass("com/qualcomm/robotcore/hardware/DcMotor$RunMode");
        jvalue reset_encoder_ticks = libcardinal::altenv_get_static_field(this->env, runmode_class, "STOP_AND_RESET_ENCODER", "com/qualcomm/robotcore/hardware/DcMotor$RunMode");
        libcardinal::altenv_call_void_instance(this->env, this->frontLeft, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &reset_encoder_ticks);
        libcardinal::altenv_call_void_instance(this->env, this->frontRight, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &reset_encoder_ticks);
        libcardinal::altenv_call_void_instance(this->env, this->backLeft, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &reset_encoder_ticks);
        libcardinal::altenv_call_void_instance(this->env, this->backRight, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &reset_encoder_ticks);
        jvalue start = libcardinal::altenv_get_static_field(this->env, runmode_class, "RUN_WITHOUT_ENCODER", "com/qualcomm/robotcore/hardware/DcMotor$RunMode");
        libcardinal::altenv_call_void_instance(this->env, this->frontLeft, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &start);
        libcardinal::altenv_call_void_instance(this->env, this->frontRight, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &start);
        libcardinal::altenv_call_void_instance(this->env, this->backLeft, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &start);
        libcardinal::altenv_call_void_instance(this->env, this->backRight, "setMode", "(Lcom/qualcomm/robotcore/hardware/DcMotor$RunMode;)V", &start);
        this->env->DeleteLocalRef(reset_encoder_ticks.l);
        this->env->DeleteLocalRef(start.l);
        this->env->DeleteLocalRef(runmode_class);
    }

    void calculateMultipliers() {
        if (this->isDead) {this->multipliers = { .fl = 0.0, .fr = 0.0, .bl = 0.0, .br = 0.0 }; return;}
        OneValPerMotor<int> encoder_ticks = this->getEncoderTicks();
#if ANCHOR == FRONT_LEFT_MOTOR
        auto anchor = (double)encoder_ticks.fl;
#elif ANCHOR == FRONT_RIGHT_MOTOR
        auto anchor = (double)encoder_ticks.fr;
#elif ANCHOR == BACK_LEFT_MOTOR
        auto anchor = (double)encoder_ticks.bl;
#elif ANCHOR == BACK_RIGHT_MOTOR
        auto anchor = (double)encoder_ticks.br;
#else
#error
#endif
        if (anchor == 0.0) {
            this->multipliers = DEFAULT_VALUES;
        } else {
            this->multipliers = {.fl=((encoder_ticks.fl) / anchor), .fr=((encoder_ticks.fr) / anchor), .bl=((encoder_ticks.bl) / anchor), .br=((encoder_ticks.br) / anchor)};
        }
    }
#endif

    [[nodiscard]] double getYaw() const {
        if (this->isDead) return 0.0;
        jobject yawPitchRollAngles = libcardinal::altenv_call_instance(this->env, this->imu, "getRobotYawPitchRollAngles", "()Lorg/firstinspires/ftc/robotcore/external/navigation/YawPitchRollAngles;", nullptr).l;
        jclass clazz = this->env->FindClass("org/firstinspires/ftc/robotcore/external/navigation/AngleUnit");
        jvalue radians = libcardinal::altenv_get_static_field(this->env, clazz, "RADIANS", "Lorg/firstinspires/ftc/robotcore/external/navigation/AngleUnit;");
        this->env->DeleteLocalRef(clazz);
        double out = (double)libcardinal::altenv_call_instance(this->env, yawPitchRollAngles, "getYaw", "(Lorg/firstinspires/ftc/robotcore/external/navigation/AngleUnit;)D", &radians).d;
        this->env->DeleteGlobalRef(yawPitchRollAngles);
        this->env->DeleteLocalRef(radians.l);
        return out;
    }

    void loop() const {
        //make sure that the robot turns off correctly
        if (this->isDead) return;
        double yaw = this->getYaw();
        //strafing
        double x = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_y", "F").f); //drive
        double y = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_x", "F").f); //strafe
        bool slow = libcardinal::altenv_get_field(this->env, this->gamepad, "left_trigger", "F").f >= 0.7f;
#if USE_DRIVER_ORIENTED
        //reorienting
        Eigen::Vector2d move;
        move << x, y;
        Eigen::Matrix2d rotate;
        rotate << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);
        Eigen::Vector2d rotatedMove = rotate * move;
        x = rotatedMove(1);
        y = rotatedMove(0);
#endif
        //turning
        double rx = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "right_stick_x", "F").f); //turn
        //swinging
#if USE_SWING
        double ry = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "right_stick_y", "F").f); //swing
        bool swingBack = libcardinal::altenv_get_field(this->env, this->gamepad, "left_bumper", "Z").z == JNI_TRUE;
        this->update(x, y, rx, ry, swingBack, slow);
#else
        this->update(x, y, rx, slow);
#endif
    }

#if USE_SWING
    void update(double drive, double strafe, double turn, double swing, bool swingBack, bool slow = false) const {
#else
    void update(double drive, double strafe, double turn, bool slow = false) const {
#endif
        if (this->isDead) return;
        //thresholding
        drive = abs(drive) > 0.6 ? -drive : 0.0;
        strafe = abs(strafe) > 0.6 ? -strafe : 0.0;
        turn = abs(turn) > 0.6 ? turn : 0.0;
        drive *= slow ? 0.2 : 1.0;
        strafe *= slow ? 0.2 : 1.0;
        turn *= slow ? 0.2 : 1.0;
#if USE_SWING
        swing = abs(swing) > 0.6 ? swing : 0.0;
        double swingFrontLeft = (-swing) / (swingBack ? 2.0 : 1.0);
        double swingFrontRight = swing / (swingBack ? 2.0 : 1.0);
        double swingBackLeft = (-swing) / (swingBack ? 1.0 : 2.0);
        double swingBackRight = swing / (swingBack ? 1.0 : 2.0);
        jvalue pfrontLeft = {.d=(left+turnLeft+swingFrontLeft) * frontLeftMultiplier};
        jvalue pfrontRight = {.d=(right+turnRight+swingFrontRight) * frontRightMultiplier};
        jvalue pbackLeft = {.d=(right+turnLeft+swingBackLeft) * backLeftMultiplier};
        jvalue pbackRight = {.d=(left+turnRight+swingBackRight) * backRightMultiplier};
#else
#if USE_DRIVER_ORIENTED
        jvalue pfrontLeft = {.d=(drive+turn-strafe)*SPEED};
        jvalue pfrontRight = {.d=(drive-turn-strafe)*SPEED};
        jvalue pbackLeft = {.d=(-drive+turn-strafe)*SPEED};
        jvalue pbackRight = {.d=(-drive-turn-strafe)*SPEED};
#else
        jvalue pfrontLeft = {.d=(drive+turn-strafe)*SPEED};
        jvalue pfrontRight = {.d=(drive-turn-strafe)*SPEED};
        jvalue pbackLeft = {.d=(drive+turn+strafe)*SPEED};
        jvalue pbackRight = {.d=(drive-turn+strafe)*SPEED};
#endif
#endif
#if USE_MOTOR_CALIBRATION
        pfrontLeft = {.d=(pfrontLeft.d * this->multipliers.fl)};
        pfrontRight = {.d=(pfrontRight.d * this->multipliers.fr)};
        pbackLeft = {.d=(pbackLeft.d * this->multipliers.bl)};
        pbackRight = {.d=(pbackRight.d * this->multipliers.br)};
#endif
        //updating
        libcardinal::altenv_call_void_instance(this->env, this->frontLeft, "setPower", "(D)V", &pfrontLeft);
        libcardinal::altenv_call_void_instance(this->env, this->frontRight, "setPower", "(D)V", &pfrontRight);
        libcardinal::altenv_call_void_instance(this->env, this->backLeft, "setPower", "(D)V", &pbackLeft);
        libcardinal::altenv_call_void_instance(this->env, this->backRight, "setPower", "(D)V", &pbackRight);
        //reset encoders for better multiplier calculation
#if USE_MOTOR_CALIBRATION
        this->calculateMultipliers();
        this->resetEncoders();
#endif
    }
};

#if USE_ARM
struct Arm {
    JNIEnv * env;
    jobject opMode;
    jobject arm_bl;
    jobject arm_br;
    //jobject arm_t;
    jobject gamepad;
    bool isDead;

    Arm(JNIEnv * env, jobject opMode) : isDead(false), env(env), opMode(env->NewGlobalRef(opMode)) {
        this->arm_bl = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "armbl", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->arm_br = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "armbr", "com/qualcomm/robotcore/hardware/DcMotor"));
        //this->arm_t = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "armt", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->gamepad = env->NewGlobalRef(libcardinal::altenv_get_field(env, opMode, "gamepad1", "Lcom/qualcomm/robotcore/hardware/Gamepad;").l);
        jclass clazz = env->FindClass("com/qualcomm/robotcore/hardware/DcMotor$ZeroPowerBehavior");
        jvalue val = libcardinal::altenv_get_static_field(env, clazz, "BRAKE", "Lcom/qualcomm/robotcore/hardware/DcMotor$ZeroPowerBehavior;");
        libcardinal::altenv_call_void_instance(env, this->arm_bl, "setZeroPowerBehavior", "(Lcom/qualcomm/robotcore/hardware/DcMotor$ZeroPowerBehavior;)V", &val);
        libcardinal::altenv_call_void_instance(env, this->arm_br, "setZeroPowerBehavior", "(Lcom/qualcomm/robotcore/hardware/DcMotor$ZeroPowerBehavior;)V", &val);
        //libcardinal::altenv_call_void_instance(env, this->arm_t, "setZeroPowerBehavior", "(Lcom/qualcomm/robotcore/hardware/DcMotor$ZeroPowerBehavior;)V", &val);
        env->DeleteLocalRef(clazz);
        env->DeleteLocalRef(val.l);
    }

    ~Arm() {
        this->env->DeleteGlobalRef(this->arm_bl);
        this->env->DeleteGlobalRef(this->arm_br);
        //this->env->DeleteGlobalRef(this->arm_t);
    }

    void log_encoders() const {
        if (this->isDead) return;
        int arm_bl_ticks = abs(libcardinal::altenv_call_instance(this->env, this->arm_bl, "getCurrentPosition", "()I", nullptr).i);
        int arm_br_ticks = abs(libcardinal::altenv_call_instance(this->env, this->arm_br, "getCurrentPosition", "()I", nullptr).i);
        //int arm_t_ticks = abs(libcardinal::altenv_call_instance(this->env, this->arm_t, "getCurrentPosition", "()I", nullptr).i);
        print("Encoder ticks: (arm_bl: %i), (arm_br: %i), (arm_t: %i)", arm_bl_ticks, arm_br_ticks, arm_t_ticks);
    }

    [[nodiscard]] jvalue * get_static() const {

    }

    void loop() const {
        if (this->isDead) return;
        bool dpad_up = libcardinal::altenv_get_field(this->env, this->gamepad, "dpad_up", "Z").z;
        bool dpad_down = libcardinal::altenv_get_field(this->env, this->gamepad, "dpad_down", "Z").z;
        //double ry = (double)(libcardinal::altenv_get_field(this->env, this->gamepad, "right_stick_y", "F").f);
        this->update(dpad_up, dpad_down/*, ry*/);
        this->log_encoders();
    }

    void update(bool dpad_up, bool dpad_down/*, double ry*/) const {
        if (this->isDead) return;
        if (dpad_up and dpad_down) {
            print("WARNING: Virtual UP and DOWN pressed at same time. Nothing will happen.");
            dpad_up = false;
            dpad_down = false;
        }
        jvalue * upref = dpad_up ? &speed : (dpad_down ? &neg_speed : &zero);
        jvalue * downref = dpad_up ? &neg_speed : (dpad_down ? &speed : &zero);
        libcardinal::altenv_call_void_instance(this->env, this->arm_bl, "setPower", "(D)V", upref);
        libcardinal::altenv_call_void_instance(this->env, this->arm_br, "setPower", "(D)V", downref);
        //libcardinal::altenv_call_void_instance(this->env, this->arm_t, "setPower", "(D)V", ry > 0.8 ? (ry > 0.0 ? const_cast<jvalue *>(&speed) : const_cast<jvalue *>(&neg_speed)) : const_cast<jvalue *>(&zero));
    }
};
#endif

#if USE_CLAW
struct Claw {
    JNIEnv * env;
    jobject opMode;
    jobject wrist;
    jobject lclaw;
    jobject rclaw;

    Claw(JNIEnv * env, jobject opMode) : env(env), opMode(opMode) {
        this->wrist = env->NewGlobalRef(libcardinal::get_device_from_hardware_map(opMode, "wrist", "com/qualcomm/robotcore/hardware/Servo"));
        this->lclaw = env->NewGlobalRef(libcardinal::get_device_from_hardware_map(opMode, "lclaw", "com/qualcomm/robotcore/hardware/Servo"));
        this->rclaw = env->NewGlobalRef(libcardinal::get_device_from_hardware_map(opMode, "rclaw", "com/qualcomm/robotcore/hardware/Servo"));
    }

    void loop() const {

    }

    void vloop(double mwrist, double mlclaw, double mrclaw) const {

    }
};
#endif

//Shorthand for the JNI function below.
void run(JNIEnv * env, jobject thiz) {
    /**
     * Our main function. Code for this season will go here.
     * Unless we need more opmodes for other reasons, to keep things tidy,
     * this should be the only opmode.
     * **/
    libcardinal::setenv(env);
    Drivetrain drivetrain(env, thiz);
#if USE_ARM
    Arm arm(env, thiz);
#endif
#if USE_CLAW
    Claw claw(env, thiz);
#endif
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    while (libcardinal::call_instance(thiz, "opModeIsActive", "()Z", nullptr).z == JNI_TRUE) {
        drivetrain.loop();
#if USE_ARM
        arm.loop();
#endif
#if USE_CLAW
        claw.loop();
#endif
    }
}

//Shorthand for the JNI function below
void runAutoBoard(JNIEnv * env, jobject thiz) {
    /**
     * Where the code for one of the autonomous modes goes.
     * **/
    libcardinal::setenv(env);
    Drivetrain drivetrain(env, thiz);
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    while (libcardinal::call_instance(thiz, "opModeIsActive", "()V", nullptr).z == JNI_TRUE) {
        //move left twice
    }
}

void runAutoStage(JNIEnv * env, jobject thiz) {
    /**
     * Where the code for one of the autonomous modes goes.
     * **/
    libcardinal::setenv(env);
    Drivetrain drivetrain(env, thiz);
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    while (libcardinal::call_instance(thiz, "waitForStart", "()V", nullptr).z == JNI_TRUE) {
        //move forwards twice, then left four or five times
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Main_runOpMode(JNIEnv * env, jobject thiz) {
    run(env, thiz);
}
extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Board_runOpMode(JNIEnv *env, jobject thiz) {
    runAutoBoard(env, thiz);
}
extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Stage_runOpMode(JNIEnv *env, jobject thiz) {
    runAutoStage(env, thiz);
}