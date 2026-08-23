"""MicroPython firmware: USB serial throttle commands to MCP4728 Channel A.

Copy this file to the Raspberry Pi Pico 2 W as ``main.py``.  The firmware starts
at HOVER and returns to HOVER if a valid packet is not received for 500 ms.
"""

from machine import I2C, Pin
import sys
import time

try:
    import uselect as select
except ImportError:
    import select


SDA_PIN = 4
SCL_PIN = 5
I2C_BUS = 0  # GP4/GP5 are hardware I2C0 SDA/SCL on Pico 2 and Pico 2 W.
I2C_FREQUENCY_HZ = 400_000
MCP4728_ADDRESS = 0x60

VREF = 3.3
HOVER_VOLTAGE = 1.65
DAC_MIN = 0
DAC_MAX = 4095

WATCHDOG_TIMEOUT_MS = 500
MAX_LINE_LENGTH = 64


def clamp_dac(value):
    return max(DAC_MIN, min(DAC_MAX, int(value)))


def voltage_to_dac(desired_voltage):
    return clamp_dac(round((desired_voltage / VREF) * DAC_MAX))


HOVER_DAC = voltage_to_dac(HOVER_VOLTAGE)


def initialize_i2c():
    bus = I2C(
        I2C_BUS,
        sda=Pin(SDA_PIN),
        scl=Pin(SCL_PIN),
        freq=I2C_FREQUENCY_HZ,
    )
    devices = bus.scan()
    if MCP4728_ADDRESS not in devices:
        raise RuntimeError(
            "MCP4728 not found at 0x{:02X}; I2C scan found {}".format(
                MCP4728_ADDRESS,
                ["0x{:02X}".format(address) for address in devices],
            )
        )
    return bus


i2c = initialize_i2c()


def set_channel_a(value):
    """Write a 12-bit value to volatile Channel A and update VA immediately.

    MCP4728 datasheet section 5.6.2 defines the Multi-Write command:

    * byte 1, 0x40: C2:C0=010 (Multi-Write), W1:W0=00,
      DAC1:DAC0=00 (Channel A), UDAC=0 (update output now)
    * byte 2: VREF=0 (VDD), PD1:PD0=00 (normal), gain=0 (1x), D11:D8
    * byte 3: D7:D0

    This changes the input/output register only; it does not write EEPROM.
    """
    safe_value = clamp_dac(value)
    command = bytes((
        0x40,
        (safe_value >> 8) & 0x0F,
        safe_value & 0xFF,
    ))
    i2c.writeto(MCP4728_ADDRESS, command)


def parse_throttle_line(line):
    """Return an in-range DAC integer, or None for any malformed packet."""
    parts = line.strip().split(",")
    if len(parts) != 2 or parts[0] != "THROTTLE":
        return None
    try:
        value = int(parts[1])
    except (TypeError, ValueError):
        return None
    # Out-of-range packets are malformed and must not update the DAC.
    if value < DAC_MIN or value > DAC_MAX:
        return None
    return clamp_dac(value)


def run():
    # Establish a defined, neutral output before accepting serial commands.
    set_channel_a(HOVER_DAC)
    print("HS210 throttle DAC ready; startup VA=HOVER ({})".format(HOVER_DAC))

    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)
    buffer = ""
    dropping_oversized_line = False
    last_valid_packet_ms = time.ticks_ms()
    watchdog_active = False
    last_watchdog_error_ms = time.ticks_ms() - 1000

    while True:
        events = poller.poll(10)
        if events:
            character = sys.stdin.read(1)
            if character:
                if character in "\r\n":
                    if not dropping_oversized_line and buffer:
                        value = parse_throttle_line(buffer)
                        if value is not None:
                            try:
                                set_channel_a(value)
                                last_valid_packet_ms = time.ticks_ms()
                                watchdog_active = False
                            except OSError as error:
                                print("MCP4728 write error:", error)
                    buffer = ""
                    dropping_oversized_line = False
                elif not dropping_oversized_line:
                    if len(buffer) < MAX_LINE_LENGTH:
                        buffer += character
                    else:
                        # Discard the whole oversized packet through its newline.
                        buffer = ""
                        dropping_oversized_line = True

        now_ms = time.ticks_ms()
        if time.ticks_diff(now_ms, last_valid_packet_ms) >= WATCHDOG_TIMEOUT_MS:
            if not watchdog_active:
                try:
                    set_channel_a(HOVER_DAC)
                    watchdog_active = True
                    print("Throttle watchdog: VA returned to HOVER")
                except OSError as error:
                    # Keep retrying until the hardware accepts the safe fallback.
                    if time.ticks_diff(now_ms, last_watchdog_error_ms) >= 1000:
                        print("Watchdog MCP4728 write error:", error)
                        last_watchdog_error_ms = now_ms


try:
    run()
except KeyboardInterrupt:
    set_channel_a(HOVER_DAC)
    print("Stopped; VA=HOVER")
