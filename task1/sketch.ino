#include <Servo.h>

const int SERVO1_PIN = 2;
const int SERVO2_PIN = 3;
const float l1 = 7.5;
const float l2 = 6.5;

Servo servo1;
Servo servo2;

void setup() {
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  Serial.begin(9600);
}

void loop() {
  if ( Serial.available () > 0) {
    String input = Serial.readString();
    int index = input.indexOf(' ');

    int first_angle = input.substring(0, index).toInt();
    int second_angle = input.substring(index).toInt();

    Serial.println(first_angle);

    servo1.write(first_angle);
    servo2.write(second_angle);
    float x1 = l1*cos(first_angle);
    float y1 = l1*sin(first_angle);
    float x_go = x1 - l2*cos(first_angle + second_angle);
    float y_go = y1 - l2*sin(first_angle + second_angle);
    String out_1 =  "x1: " + String(x1) + "   y1: " + String(y1);
    String out_2 =  "x2: " + String(x_go) + "   y2: " + String(y_go);
    Serial.println(out_1);
    Serial.println(out_2);
  }
}