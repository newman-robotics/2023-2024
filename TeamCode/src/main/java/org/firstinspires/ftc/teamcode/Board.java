package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.Autonomous;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;

@Autonomous(name="Board")
public class Board extends LinearOpMode {
    static {
        System.loadLibrary("main");
    }

    public native void runOpMode() throws InterruptedException;
}
