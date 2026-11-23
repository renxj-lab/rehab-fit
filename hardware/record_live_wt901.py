import asyncio
import struct
import time
import os
import csv
from bleak import BleakClient


# =====================================================
# WT901BLE SETTINGS
# =====================================================
ADDRESS = "E1:05:3D:86:3A:03"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

SAVE_FOLDER = "live_training_data"

VALID_LABELS = ["ABD", "FEL", "IR", "PEN", "REST", "WRONG"]


# =====================================================
# WT901BLE PACKET PARSER
# =====================================================
def parse_wt901_packet(packet):
    """
    WT901BLE BLE packet format:
    20 bytes:
    55 61 ax ay az wx wy wz angle_x angle_y angle_z
    """

    if len(packet) != 20:
        return None

    if packet[0] != 0x55 or packet[1] != 0x61:
        return None

    try:
        values = struct.unpack("<hhhhhhhhh", packet[2:20])
    except Exception:
        return None

    ax_raw, ay_raw, az_raw, wx_raw, wy_raw, wz_raw, roll_raw, pitch_raw, yaw_raw = values

    ax = ax_raw / 32768.0 * 16.0
    ay = ay_raw / 32768.0 * 16.0
    az = az_raw / 32768.0 * 16.0

    wx = wx_raw / 32768.0 * 2000.0
    wy = wy_raw / 32768.0 * 2000.0
    wz = wz_raw / 32768.0 * 2000.0

    angle_x = roll_raw / 32768.0 * 180.0
    angle_y = pitch_raw / 32768.0 * 180.0
    angle_z = yaw_raw / 32768.0 * 180.0

    return {
        "time": time.time(),
        "ax": ax,
        "ay": ay,
        "az": az,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "angle_x": angle_x,
        "angle_y": angle_y,
        "angle_z": angle_z,
    }


# =====================================================
# RECORD DATA
# =====================================================
async def record_data(label, duration, output_file):
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    filepath = os.path.join(SAVE_FOLDER, output_file)

    rows = []
    buffer = bytearray()

    print()
    print("Connecting to WT901BLE...")
    print("Close WitMotion software, FastAPI backend, Streamlit, and dashboard first.")
    print()

    def notification_handler(sender, data):
        nonlocal buffer, rows

        buffer.extend(data)

        while len(buffer) >= 20:
            if not (buffer[0] == 0x55 and buffer[1] == 0x61):
                buffer.pop(0)
                continue

            packet = buffer[:20]
            del buffer[:20]

            sample = parse_wt901_packet(packet)

            if sample is not None:
                sample["label"] = label
                rows.append(sample)

    try:
        async with BleakClient(ADDRESS) as client:
            print("Connected to WT901BLE.")
            print()
            print(f"Label      : {label}")
            print(f"Duration   : {duration} seconds")
            print(f"Save file  : {filepath}")
            print()
            print("Get into the correct starting position now.")
            print("Recording starts in 3 seconds...")
            print()

            await asyncio.sleep(3)

            await client.start_notify(NOTIFY_UUID, notification_handler)

            start_time = time.time()

            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(
                    f"Recording {label}... {remaining:5.1f} seconds left | Samples: {len(rows)}",
                    end="\r",
                )
                await asyncio.sleep(0.2)

            await client.stop_notify(NOTIFY_UUID)

    except Exception as e:
        print()
        print("ERROR while recording:")
        print(e)
        print()
        print("Check these:")
        print("1. Is WT901BLE switched on?")
        print("2. Is Bluetooth on?")
        print("3. Is WitMotion app closed?")
        print("4. Is FastAPI backend closed?")
        print("5. Is another script already connected?")
        return

    print()
    print()
    print("Recording finished.")
    print(f"Samples collected: {len(rows)}")

    if len(rows) == 0:
        print("No data was recorded. Try again.")
        return

    fieldnames = [
        "time",
        "ax",
        "ay",
        "az",
        "wx",
        "wy",
        "wz",
        "angle_x",
        "angle_y",
        "angle_z",
        "label",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved successfully to: {filepath}")
    print()


# =====================================================
# USER INPUT
# =====================================================
def show_recording_instructions(label):
    print()
    print("Recording instructions:")
    print("-----------------------")

    if label == "ABD":
        print("ABD = Shoulder Abduction")
        print("Arm starts down beside body.")
        print("Lift arm sideways, then return down.")
        print("Do slow clean reps.")

    elif label == "FEL":
        print("FEL = Forward Elevation")
        print("Arm starts down.")
        print("Lift arm forward, then return down.")
        print("Do slow clean reps.")

    elif label == "IR":
        print("IR = Internal Rotation")
        print("Keep elbow bent.")
        print("Rotate arm inward, then return.")
        print("Do slow controlled rotations.")

    elif label == "PEN":
        print("PEN = Pendulum")
        print("Arm relaxed.")
        print("Do gentle pendulum swings.")
        print("Do not make large random movements.")

    elif label == "REST":
        print("REST = No exercise")
        print("Stay still in starting position.")
        print("Do not perform any exercise.")
        print("Small natural body movement is okay.")

    elif label == "WRONG":
        print("WRONG = Incorrect movement")
        print("Do bad/random movements:")
        print("- tiny half reps")
        print("- wrong direction")
        print("- shaking arm")
        print("- moving too fast")
        print("- mixing exercises")
        print("- not returning to start")

    print()


def main():
    print("==========================================")
    print("WT901BLE LIVE TRAINING DATA RECORDER")
    print("==========================================")
    print()
    print("Available labels:")
    print("ABD   = Shoulder Abduction")
    print("FEL   = Forward Elevation")
    print("IR    = Internal Rotation")
    print("PEN   = Pendulum")
    print("REST  = No exercise / still")
    print("WRONG = Incorrect movement")
    print()

    label = input("Enter label: ").strip().upper()

    if label not in VALID_LABELS:
        print()
        print("Invalid label.")
        print(f"Use one of these: {', '.join(VALID_LABELS)}")
        return

    duration_input = input("Enter duration in seconds, example 25: ").strip()

    try:
        duration = int(duration_input)
    except ValueError:
        print("Invalid duration. Use a number like 25.")
        return

    filename = input("Enter output filename, example ABD_live_3.csv: ").strip()

    if filename == "":
        filename = f"{label}_live_{int(time.time())}.csv"

    if not filename.endswith(".csv"):
        filename += ".csv"

    show_recording_instructions(label)

    confirm = input("Ready to record? Type y to start: ").strip().lower()

    if confirm != "y":
        print("Recording cancelled.")
        return

    asyncio.run(record_data(label, duration, filename))


if __name__ == "__main__":
    main()
