# Hand Gesture Controlled Drone

A hand-gesture-controlled drone system that uses **computer vision, embedded hardware, and digital-to-analog conversion** to control the throttle of a Holy Stone HS210 drone.

The system detects the user's hand through a webcam and classifies the gesture as:

* **Hand Up → CLIMB**
* **Fist / Neutral → HOVER**
* **Hand Down → DESCEND**

The detected gesture is converted into a throttle command and sent to a Raspberry Pi Pico 2 W, which controls an MCP4728 DAC connected to the drone transmitter.

## DEMO
[C:\Users\afraz\AppData\Local\CapCut\Videos]

---

## Project Overview

The goal of this project was to replace manual throttle movement with **real-time hand gesture control**.

A webcam continuously tracks the user's hand. The computer determines whether the fingers are pointing upward, downward, or are in a neutral/closed position.

That gesture is converted into one of three throttle levels:

| Gesture               | Command | Output Voltage |
| --------------------- | ------- | -------------: |
| Fingers pointing up   | CLIMB   |         3.30 V |
| Fist / Neutral        | HOVER   |         1.65 V |
| Fingers pointing down | DESCEND |         0.00 V |

Only the **throttle axis** is controlled by the gesture system.

Pitch, roll, yaw, pairing, and arming remain controlled by the original HS210 transmitter.

---

## How the System Works

```text
Hand Gesture
     ↓
Computer Webcam
     ↓
OpenCV + MediaPipe
     ↓
Gesture Detection
     ↓
CLIMB / HOVER / DESCEND
     ↓
USB Serial Communication
     ↓
Raspberry Pi Pico 2 W
     ↓
MCP4728 DAC
     ↓
Analog Throttle Voltage
     ↓
HS210 Transmitter
     ↓
Drone
```

### Hand Detection

The computer webcam captures the user's hand in real time.

MediaPipe identifies the finger positions and determines whether the hand is pointing up, down, or is in a neutral/closed position.

### Throttle Control

The gesture is converted into a digital throttle value and sent to the Raspberry Pi Pico 2 W.

The Pico communicates with the MCP4728 DAC using I2C.

The DAC converts the digital command into the analog voltage used by the HS210 transmitter's throttle input.

| Drone State | Voltage | DAC Output |
| ----------- | ------: | ---------: |
| DESCEND     |  0.00 V |          0 |
| HOVER       |  1.65 V |       2048 |
| CLIMB       |  3.30 V |       4095 |

---

## Bill of Materials

| Component              | Quantity | Purpose                                       |
| ---------------------- | -------: | --------------------------------------------- |
| Holy Stone HS210 Drone |        1 | Drone being controlled                        |
| HS210 Transmitter      |        1 | Sends commands to the drone                   |
| Raspberry Pi Pico 2 W  |        1 | Embedded controller                           |
| MCP4728 DAC            |        1 | Converts digital commands into analog voltage |
| Breadboard             |        1 | Circuit prototyping                           |
| Jumper Wires           |  Several | Electrical connections                        |
| Computer / Laptop      |        1 | Runs hand-tracking software                   |
| Webcam                 |        1 | Detects hand gestures                         |
| USB Data Cable         |        1 | Computer-to-Pico communication and power      |
| Digital Multimeter     |        1 | Measures and verifies throttle voltages       |

---

## Circuit Connections

### Raspberry Pi Pico 2 W to MCP4728

| Raspberry Pi Pico 2 W | MCP4728   |
| --------------------- | --------- |
| GP4                   | SDA       |
| GP5                   | SCL       |
| 3V3                   | VCC / VIN |
| GND                   | GND       |

### MCP4728 to HS210 Transmitter

| MCP4728        | HS210 Transmitter    |
| -------------- | -------------------- |
| VA / Channel A | Throttle Wiper Input |
| GND            | Controller Ground    |

The Pico, MCP4728, and HS210 transmitter must share a **common ground**.

The transmitter remains powered by its original batteries, while the Raspberry Pi Pico is powered through USB.

---

## Safety System

If the system is unsure what command to send, it defaults to **HOVER**.

HOVER is selected when:

* No hand is detected
* The gesture is unclear
* Detection confidence is too low
* Serial communication is lost
* The desktop program stops communicating

The Raspberry Pi Pico also uses an independent watchdog. If valid commands stop arriving for approximately **500 ms**, the system returns the DAC output to the HOVER voltage.

---

## How to Run the Code

### 1. Install Python

Python **3.11** is recommended.

Check your installation:

```powershell
py -3.11 --version
```

### 2. Create a Virtual Environment

Open PowerShell inside the project folder:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Test the Webcam First

Run the project without connecting the Pico:

```powershell
python hand_throttle.py --no-serial
```

This allows you to verify that the webcam and gesture detection are working correctly.

Press **Q** or **ESC** to exit.

### 4. Install MicroPython on the Pico 2 W

Flash the Raspberry Pi Pico 2 W with MicroPython.

Then install `mpremote`:

```powershell
pip install mpremote
```

Check that the Pico is detected:

```powershell
mpremote connect list
```

### 5. Upload the Pico Controller

Upload the Pico code as `main.py`:

```powershell
mpremote cp pico/pico_dac_controller.py :main.py
```

Reset the Pico:

```powershell
mpremote reset
```

### 6. Verify the MCP4728

The MCP4728 should appear at I2C address:

```text
0x60
```

Run:

```powershell
mpremote connect auto exec "from machine import Pin, I2C; i2c=I2C(0, sda=Pin(4), scl=Pin(5), freq=400000); print([hex(address) for address in i2c.scan()])"
```

Expected result:

```text
['0x60']
```

### 7. Calibrate the Output

Before connecting the DAC output to the HS210 transmitter, verify the voltage with a multimeter.

Run:

```powershell
python hand_throttle.py --calibrate --port COM5
```

Replace `COM5` with the Pico's actual COM port.

Expected outputs:

| Command | Voltage |
| ------- | ------: |
| DESCEND | ~0.00 V |
| HOVER   | ~1.65 V |
| CLIMB   | ~3.30 V |

### 8. Run the Full System

Once the camera and voltage outputs have been verified:

```powershell
python hand_throttle.py --port COM5
```

Or, if automatic port detection works:

```powershell
python hand_throttle.py
```

The program will open the webcam and begin converting hand gestures into throttle commands.

---

## Authors

**Afraz Jajja**
Electrical Engineering

**OpenAI Codex**
AI-assisted software development

**Project Areas:**
Computer Vision • Embedded Systems • Electronics • Control Systems
