package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

@TeleOp(name="Main. Our main tele-op.")
public class Main extends LinearOpMode {
    static {
        System.loadLibrary("main");
    }

    public native void runOpMode() throws InterruptedException;
}
