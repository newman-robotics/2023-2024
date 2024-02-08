#include <memory>
#include <optional>
#include <ctime>
#include <jni.h>
#include "libcardinal.hpp"
#include <apriltags/TagDetector.h>
#include <android/log.h>
#include <cstdlib>

//how we implement print(...)
#define  LOGGER_NAME "El J.U.L.I.O. L.I."
#define  print(...) __android_log_print(ANDROID_LOG_INFO,LOGGER_NAME,__VA_ARGS__)

enum MotorLocation {
    FRONT_LEFT_MOTOR,
    FRONT_RIGHT_MOTOR,
    BACK_LEFT_MOTOR,
    BACK_RIGHT_MOTOR,
};

enum ArmTarget {
    TARGET_OUT = 0,
    TARGET_MID = 1,
    TARGET_IN = 2
};

//compilation params

//whether to use driver oriented driving
#define  USE_DRIVER_ORIENTED false

//whether to use the jules
//Jules is stupid and doesn't know how to make a functional jules,
//so don't turn this on unless trying (in vain) to calibrate the jules
#define  USE_JULES false

//which one of these two uses of the right joystick should be used
//do NOT EVER enable both of them, or weird things will happen
//however, they can be both disabled
#define  USE_SWING false
#define  USE_ARM true

//gamepad controls
#define  INVERT_X true
#define  INVERT_Y true

//certain other things
//that do things
#define  USE_MOTOR_CALIBRATION false
#define  USE_CLAW true
#define  USE_CTC true

//which motor is the anchor, i.e. which motor should every other
//motor's ticks be pinned to
#define  ANCHOR FRONT_LEFT_MOTOR

//the speed multiplier of the motors
//0.0 is none, 1.0 is full
#define  SPEED 0.9
#define  ARM_SPEED 0.5

//where the servo motor's open and closed positions are
#define  LSERVO_OPEN 0.5
#define  LSERVO_CLOSE 0.0
#define  RSERVO_OPEN 0.0
#define  RSERVO_CLOSE 0.4

//Jules settings

#define  JULES_P 0.8
#define  JULES_I 0.0
#define  JULES_D 0.2

#define  INTERNAL_TARGET_OUT 100 //??
#define  INTERNAL_TARGET_MID 50 //??
#define  INTERNAL_TARGET_IN 0 //??

//CTC settings

#define  CTC_START 0
#define  CTC_END 10000

static jvalue speed = {.d=ARM_SPEED};
static jvalue zero = {.d=0.0};
static jvalue neg_speed = {.d=-ARM_SPEED};

static jvalue ctc_start = {.i=CTC_START};
static jvalue ctc_end = {.i=CTC_END};

static inline double toRadians(double degrees) {
    return (degrees * M_PI) / 180.0;
}

static inline double toDegrees(double radians) {
    return (radians * 180.0) / M_PI;
}

#if USE_JULES
static inline int getTargetTicks(ArmTarget target) {
    switch (target) {
        case TARGET_OUT: return INTERNAL_TARGET_OUT;
        case TARGET_MID: return INTERNAL_TARGET_MID;
        case TARGET_IN: return INTERNAL_TARGET_IN;
        default: {
            print("Got bad ArmTarget: %i", target);
            //put some sort of java error handling wizardry here
            //or a POSIX signal or something
            //or just...
            std::exit(-1);
            //yaaaay!
        }
    }
}
#endif

long micro_time() {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return tv.tv_usec + tv.tv_sec * 1000000;
}

template <typename T> struct OneValPerMotor {
    T fl;
    T fr;
    T bl;
    T br;
};

const OneValPerMotor<double> DEFAULT_VALUES = {.fl=1.0,.fr=1.0,.bl=1.0,.br=1.0};

#if USE_JULES
struct Jules {
    explicit Jules(double K_p = 0.0, double K_i = 0.0, double K_d = 0.0, std::optional<double> integral_limit = std::optional<double>()) : K_p(K_p), K_i(K_i), K_d(K_d), integral_limit(integral_limit) {
        this->reset();
    }

    void reset() {
        this->last_error = 0.0;
        this->integral = 0.0;
        this->last_time = micro_time();
    }

    double operator() (double error, std::optional<double> error_derivative = {}) {
        long current_time = micro_time();
        long dt = current_time - this->last_time;
        if (dt == 0.0) return 0.0;
        this->last_time = current_time;
        this->integral = this->_get_integral(error, dt);
        double derivative = error_derivative ? error_derivative.value() : this->_get_derivative(error, dt);
        double output = this->K_p * error + this->K_i * this->integral + this->K_d * derivative;
        this->last_error = error;
        return output;
    }

private:
    double K_p, K_i, K_d, last_error{}, integral{};
    long last_time{};
    std::optional<double> integral_limit;

    [[nodiscard]] double _get_integral(double error, double dt) const {
        double _integral = this->integral + error * dt;
        if (this->integral_limit) {
            _integral = max(-integral_limit.value(), min(integral_limit.value(), _integral));
        }
        return _integral;
    }

    [[nodiscard]] double _get_derivative(double error, double dt) const {
        return (error - this->last_error) / dt;
    }
};
#endif

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
        this->frontLeft = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivefl", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->frontRight = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivefr", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->backLeft = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivebl", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->backRight = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "drivebr", "com/qualcomm/robotcore/hardware/DcMotor"));
        this->imu = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "imu", "com/qualcomm/robotcore/hardware/IMU"));
        this->gamepad = env->NewGlobalRef(libcardinal::altenv_get_field(env, opMode, "gamepad1", "Lcom/qualcomm/robotcore/hardware/Gamepad;").l);
        //jvalue reversed = libcardinal::altenv_get_static_field(env, env->FindClass("com/qualcomm/robotcore/hardware/DcMotorSimple$Direction"), "REVERSE", "Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;");
        //libcardinal::call_void_instance(this->frontLeft, "setDirection", "(Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;)V", &reversed);
        //libcardinal::call_void_instance(this->frontRight, "setDirection", "(Lcom/qualcomm/robotcore/hardware/DcMotorSimple$Direction;)V", &reversed);
        libcardinal::call_void_instance(this->imu, "resetYaw", "()V", nullptr);
    }
    ~Drivetrain() {
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

    void setTarget(long ticks) const {}

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
        //double yaw = this->getYaw();
        //strafing
#if INVERT_X
        double x = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_y", "F").f); //drive
#else
        double x = (double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_y", "F").f); //drive
#endif
#if INVERT_Y
        double y = -(double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_x", "F").f); //strafe
#else
        double y = (double)(libcardinal::altenv_get_field(this->env, this->gamepad, "left_stick_x", "F").f); //strafe
#endif
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
        double rx = (double)(libcardinal::altenv_get_field(this->env, this->gamepad, "right_stick_x", "F").f); //turn
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
        jvalue pfrontLeft = {.d=(drive-turn-strafe)*SPEED};
        jvalue pfrontRight = {.d=(-drive-turn-strafe)*SPEED};
        jvalue pbackLeft = {.d=(drive-turn+strafe)*SPEED};
        jvalue pbackRight = {.d=(drive+turn-strafe)*SPEED};
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
        print("backLeft: %li, backRight, %li", libcardinal::altenv_call_instance(this->env, this->backLeft, "getCurrentPosition", "()I", nullptr).i, libcardinal::altenv_call_instance(this->env, this->backRight, "getCurrentPosition", "()I", nullptr).i);
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
#if USE_JULES
    Jules jules;
    ArmTarget target;
#endif

    Arm(JNIEnv * env, jobject opMode) : isDead(false), env(env), opMode(env->NewGlobalRef(opMode))
#if USE_JULES
    , jules(JULES_P, JULES_I, JULES_D), target(TARGET_IN)
#endif
    {
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

    void log_encoders() const {}

    void loop() {
        if (this->isDead) return;
        bool dpad_up = libcardinal::altenv_get_field(this->env, this->gamepad, "dpad_up", "Z").z;
        bool dpad_down = libcardinal::altenv_get_field(this->env, this->gamepad, "dpad_down", "Z").z;
        //double ry = (double)(libcardinal::altenv_get_field(this->env, this->gamepad, "right_stick_y", "F").f);
        this->update(dpad_up, dpad_down/*, ry*/);
        this->log_encoders();
    }

    void update(bool dpad_up, bool dpad_down/*, double ry*/) {
        if (this->isDead) return;
        if (dpad_up and dpad_down) {
            print("WARNING: Virtual UP and DOWN pressed at same time. Nothing will happen.");
            dpad_up = false;
            dpad_down = false;
        }
#if USE_JULES
        if (dpad_up) {
            this->target = (ArmTarget)((int)this->target + 1);
            if ((int)this->target > 2) this->target = (ArmTarget)2;
        } else if (dpad_down) {
            this->target = (ArmTarget)((int)this->target - 1);
            if ((int)this->target < 0) this->target = (ArmTarget)0;
        }
        jvalue power = {.d=this->jules(getTargetTicks(this->target) - libcardinal::altenv_call_instance(this->env, this->arm_bl,"getCurrentPosition", "()I",nullptr).i)};
        jvalue negpower = {.d=-power.d};
        libcardinal::altenv_call_void_instance(this->env, this->arm_bl, "setPower", "(D)V", &power);
        libcardinal::altenv_call_void_instance(this->env, this->arm_br, "setPower", "(D)V", &negpower);
#else
        jvalue * upref = dpad_up ? &speed : (dpad_down ? &neg_speed : &zero);
        jvalue * downref = dpad_up ? &neg_speed : (dpad_down ? &speed : &zero);
        libcardinal::altenv_call_void_instance(this->env, this->arm_bl, "setPower", "(D)V", upref);
        libcardinal::altenv_call_void_instance(this->env, this->arm_br, "setPower", "(D)V", downref);
#endif
    }
};
#endif

#if USE_CLAW
struct Claw {
    JNIEnv * env;
    jobject opMode;
    jobject lclaw;
    jobject rclaw;
    jobject gamepad;

    Claw(JNIEnv * env, jobject opMode) : env(env), opMode(opMode) {
        print("CLAW INIT");
        this->lclaw = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "lclaw", "com/qualcomm/robotcore/hardware/Servo"));
        this->rclaw = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "rclaw", "com/qualcomm/robotcore/hardware/Servo"));
        this->gamepad = env->NewGlobalRef(libcardinal::altenv_get_field(env, opMode, "gamepad1", "Lcom/qualcomm/robotcore/hardware/Gamepad;").l);
    }

    ~Claw() {
        this->env->DeleteGlobalRef(this->lclaw);
        this->env->DeleteGlobalRef(this->rclaw);
        this->env->DeleteGlobalRef(this->gamepad);
    }

    void loop() const {
        bool open = libcardinal::altenv_get_field(this->env, this->gamepad, "right_bumper", "Z").z == JNI_TRUE;
        double lposition = open ? LSERVO_OPEN : LSERVO_CLOSE;
        double rposition = open ? RSERVO_OPEN : RSERVO_CLOSE;
        this->update(lposition, rposition);
    }

    void update(double mlclaw, double mrclaw) const {
        jvalue mlclaw_v = {.d=mlclaw};
        jvalue mrclaw_v = {.d=mrclaw};
        libcardinal::altenv_call_void_instance(this->env, this->lclaw, "setPosition", "(D)V", &mlclaw_v);
        libcardinal::altenv_call_void_instance(this->env, this->rclaw, "setPosition", "(D)V", &mrclaw_v);
    }
};
#endif

#if USE_CTC
struct CTC {
    JNIEnv * env;
    jobject opMode;
    jobject launcher;
    jobject gamepad;

    CTC(JNIEnv * env, jobject opMode) : env(env), opMode(opMode) {
        this->launcher = env->NewGlobalRef(libcardinal::altenv_get_device_from_hardware_map(env, opMode, "yuta", "com/qualcomm/robotcore/hardware/Servo"));
        this->gamepad = env->NewGlobalRef(libcardinal::altenv_get_field(env, opMode, "gamepad1", "Lcom/qualcomm/robotcore/hardware/Gamepad;").l);
        libcardinal::altenv_call_void_instance(this->env, this->launcher, "setTargetPosition", "(I)V", &ctc_start);
    }

    ~CTC() {
        this->env->DeleteGlobalRef(this->launcher);
        this->env->DeleteGlobalRef(this->gamepad);
    }

    void loop() const {
        bool launch = libcardinal::altenv_get_field(this->env, this->gamepad, "square", "Z").z == JNI_TRUE;
        if (launch) {
            libcardinal::altenv_call_void_instance(this->env, this->launcher, "setTargetPosition","(I)V", &ctc_end);
        }
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
    print("Size test: char = %i, short = %i, int = %i, long = %i, long long = %i", sizeof(char), sizeof(short), sizeof(int), sizeof(long), sizeof(long long));
    Drivetrain drivetrain(env, thiz);
#if USE_ARM
    Arm arm(env, thiz);
#endif
#if USE_CLAW
    Claw claw(env, thiz);
#endif
#if USE_CTC
    CTC ctc(env, thiz);
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
#if USE_CTC
        ctc.loop();
#endif
    }
}

#if USE_AUTO

enum AutoControllerInstruction {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    CLOCKWISE,
    COUNTERCLOCKWISE,
    DROP,
    UP,
    DOWN
};

struct AutoController {
    JNIEnv * env;
    jobject opMode;
    Drivetrain& drivetrain;
    Arm& arm;
    Claw& claw;

    AutoController(JNIEnv * env, jobject opMode, Drivetrain& drivetrain, Arm& arm, Claw& claw) : env(env), opMode(opMode), drivetrain(drivetrain), arm(arm), claw(claw) {}
    ~AutoController() = default;

    void forward(int tiles) {queue.emplace_back(FORWARD, tiles);}
    void backward(int tiles) {queue.emplace_back(BACKWARD, tiles);}
    void left(int tiles) {queue.emplace_back(LEFT, tiles);}
    void right(int tiles) {queue.emplace_back(RIGHT, tiles);}
    void clockwise(double degrees) {queue.emplace_back(CLOCKWISE, degrees);}
    void counterclockwise(double degrees) {queue.emplace_back(COUNTERCLOCKWISE, degrees);}
    void drop() {queue.emplace_back(DROP, 0);}
    void down() {queue.emplace_back(DOWN, 0);}
    void up() {queue.emplace_back(UP, 0);}

    void step() {
        switch (this->queue[this->instruction].first) {
            case FORWARD: {
                if (!this->has_target) {
                    this->targetTicks = -AutoController::ticksPerTile;
                    this->has_target = true;
                }
                break;
            }
            case BACKWARD: {break;}
            case LEFT: {break;}
            case RIGHT: {break;}
            case CLOCKWISE: {break;}
            case COUNTERCLOCKWISE: {break;}
            case DROP: {break;}
            case UP: {break;}
            case DOWN: {break;}
            default: return;
        }
    }

private:
    std::vector<std::pair<AutoControllerInstruction, int>> queue;
    int instruction;
    bool has_target;
    long targetTicks;
    static const long ticksPerTile = 435;
    static const long ticksToRotate = 800;
};

//Shorthand for the JNI function below
void runAutoBoard(JNIEnv * env, jobject thiz) {
    /**
     * Where the code for one of the autonomous modes goes.
     * **/
    libcardinal::setenv(env);
    Drivetrain drivetrain(env, thiz);
    Arm arm(env, thiz);
    Claw claw(env, thiz);
    AutoController controller(env, thiz, drivetrain, arm, claw);
    controller.forward(1);
    controller.counterclockwise(90);
    controller.forward(1);
    controller.drop();
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    while (libcardinal::call_instance(thiz, "opModeIsActive", "()V", nullptr).z == JNI_TRUE) {
        controller.step();
    }
}

void runAutoStage(JNIEnv * env, jobject thiz) {
    /**
     * Where the code for one of the autonomous modes goes.
     * **/
    libcardinal::setenv(env);
    Drivetrain drivetrain(env, thiz);
    Arm arm(env, thiz);
    Claw claw(env, thiz);
    AutoController controller(env, thiz, drivetrain, arm, claw);
    controller.forward(2);
    controller.down();
    controller.left(3);
    controller.up();
    controller.counterclockwise(90);
    controller.left(1);
    controller.drop();
    libcardinal::call_void_instance(thiz, "waitForStart", "()V", nullptr);
    while (libcardinal::call_instance(thiz, "waitForStart", "()V", nullptr).z == JNI_TRUE) {
        controller.step();
    }
}

#endif

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Main_runOpMode(JNIEnv * env, jobject thiz) {
    run(env, thiz);
}

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Board_runOpMode(JNIEnv *env, jobject thiz) {
#if USE_AUTO
    runAutoBoard(env, thiz);
#endif
}

extern "C" JNIEXPORT void JNICALL Java_org_firstinspires_ftc_teamcode_Stage_runOpMode(JNIEnv *env, jobject thiz) {
#if USE_AUTO
    runAutoStage(env, thiz);
#endif
}