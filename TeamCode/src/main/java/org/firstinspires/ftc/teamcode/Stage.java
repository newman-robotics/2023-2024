package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.Autonomous;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;

@Autonomous(name="Stage")
public class Stage extends LinearOpMode {
    static {
        System.loadLibrary("main");
    }

    public native void runOpMode() throws InterruptedException;
}
