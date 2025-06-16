#include <Arduino.h>
#include <PS2X_lib.h>

#ifndef PS2_TYPE_DUALSHOCK
  #define PS2_TYPE_DUALSHOCK    0x03
  #define PS2_TYPE_GUITARHERO   0x01
  #define PS2_TYPE_KONAMI       0x02
  #define PS2_TYPE_UNKNOWN      0x00
#endif

// PS2 pin (TIDAK DIUBAH SESUAI PERMINTAAN)
#define PS2_DAT 13 // D7
#define PS2_CMD 12 // D6
#define PS2_SEL 4 // D2
#define PS2_CLK 14 // D5

// Motor Kiri
#define MOTOR_L_IN1  16  // RPWM
#define MOTOR_L_IN2  5  // LPWM

// Motor Kanan
#define MOTOR_R_IN1  0  // RPWM
#define MOTOR_R_IN2  2  // LPWM

#define PS2_READ_DELAY 100
const int ANALOG_DEAD_ZONE = 10;

PS2X ps2x;
int error = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("=== PS2 Controller Test (ESP32 Lolin S3) ===");
  Serial.println("Pastikan kontroler PS2 dalam mode Analog (lampu merah menyala)");

  error = ps2x.config_gamepad(
    PS2_CLK, PS2_CMD, PS2_SEL, PS2_DAT,
    true, true
  );

  if (error == 0) {
    Serial.println("Koneksi PS2 Controller BERHASIL!");
    byte type = ps2x.Analog(1);
    Serial.print("Tipe Kontroler: ");
    switch (type) {
      case PS2_TYPE_DUALSHOCK: Serial.println("DualShock"); break;
      case PS2_TYPE_GUITARHERO: Serial.println("Guitar Hero"); break;
      case PS2_TYPE_KONAMI: Serial.println("Konami DancePad"); break;
      default: Serial.println("Tidak Dikenal");
    }
  } else {
    Serial.print("Gagal menghubungkan PS2 Controller. Error code: ");
    Serial.println(error);
    if (error == 1) Serial.println("=> Kontroler tidak terdeteksi.");
    else if (error == 2) Serial.println("=> Mode analog tidak aktif.");
    else if (error == 3) Serial.println("=> Kontroler disconnect.");
  }

  // Inisialisasi pin motor sebagai output
  pinMode(MOTOR_L_IN1, OUTPUT);
  pinMode(MOTOR_L_IN2, OUTPUT);
  pinMode(MOTOR_R_IN1, OUTPUT);
  pinMode(MOTOR_R_IN2, OUTPUT);
}

// float speedMultiplier = 1.0;  // Default: kecepatan normal

void loop() {
  if (error == 0) {
    ps2x.read_gamepad(false, 0);

    // Baca analog kiri
    int8_t lx = ps2x.Analog(PSS_LX) - 128;
    int8_t ly = ps2x.Analog(PSS_LY) - 128;

    if (abs(lx) < ANALOG_DEAD_ZONE) lx = 0;
    if (abs(ly) < ANALOG_DEAD_ZONE) ly = 0;

    // Print data ke Serial
    Serial.print("Analog Kiri: X=");
    Serial.print(lx);
    Serial.print(" | Y=");
    Serial.println(ly);

    // Hitung nilai maksimum dari analog untuk normalisasi
    float magnitude = sqrt(lx * lx + ly * ly);
    if (magnitude > 127) magnitude = 127;

    // Hitung arah motor (differential drive logic)
    float leftInput  = (float)ly + lx;
    float rightInput = (float)ly - lx;

    // Normalisasi input ke -1.0 sampai 1.0
    float maxInput = max(abs(leftInput), abs(rightInput));
    if (maxInput == 0) maxInput = 1; // hindari pembagian nol
    leftInput  = (leftInput / maxInput) * (magnitude / 127.0);
    rightInput = (rightInput / maxInput) * (magnitude / 127.0);

    // Skala ke -255 sampai 255
    int leftSpeed  = int(leftInput * 255);
    int rightSpeed = int(rightInput * 255);

    // Gerakkan motor
    driveMotor(MOTOR_L_IN1, MOTOR_L_IN2, leftSpeed);
    driveMotor(MOTOR_R_IN1, MOTOR_R_IN2, rightSpeed);
  } else {
    Serial.println("Mencoba koneksi ulang ke PS2 Controller...");
    error = ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_SEL, PS2_DAT, true, true);
  }

  delay(PS2_READ_DELAY);
}


void driveMotor(int in1, int in2, int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    analogWrite(in1, speed);
    analogWrite(in2, 0);
  } else if (speed < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, -speed);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}
