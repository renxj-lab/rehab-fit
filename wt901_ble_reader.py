import asyncio
import struct
from bleak import BleakClient

ADDRESS = "E1:05:3D:86:3A:03"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

buffer = bytearray()


def parse_55_61_packet(packet):
    """
    WT901BLE combined BLE packet format:
    20 bytes:
    55 61
    ax ay az
    wx wy wz
    angle_x angle_y angle_z

    Each value is signed int16, little endian.
    """

    if len(packet) != 20:
        return None

    if packet[0] != 0x55 or packet[1] != 0x61:
        return None

    try:
        values = struct.unpack("<hhhhhhhhh", packet[2:20])
    except:
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
        "ax": ax,
        "ay": ay,
        "az": az,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "angle_x": angle_x,
        "angle_y": angle_y,
        "angle_z": angle_z
    }


def notification_handler(sender, data):
    global buffer

    buffer.extend(data)

    while len(buffer) >= 20:
        # Search for packet start: 55 61
        if not (buffer[0] == 0x55 and buffer[1] == 0x61):
            buffer.pop(0)
            continue

        packet = buffer[:20]
        del buffer[:20]

        parsed = parse_55_61_packet(packet)

        if parsed is not None:
            print(
                f"ax={parsed['ax']:.3f}, "
                f"ay={parsed['ay']:.3f}, "
                f"az={parsed['az']:.3f}, "
                f"wx={parsed['wx']:.3f}, "
                f"wy={parsed['wy']:.3f}, "
                f"wz={parsed['wz']:.3f}, "
                f"angle_x={parsed['angle_x']:.2f}, "
                f"angle_y={parsed['angle_y']:.2f}, "
                f"angle_z={parsed['angle_z']:.2f}"
            )


async def main():
    print("Connecting to WT901BLE68...")
    print("Address:", ADDRESS)

    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)

        await client.start_notify(NOTIFY_UUID, notification_handler)

        print("Started notification on:", NOTIFY_UUID)
        print("\nReading live WT901BLE data...")
        print("Move the sensor now.")
        print("Press CTRL + C to stop.\n")

        while True:
            await asyncio.sleep(1)


asyncio.run(main())
