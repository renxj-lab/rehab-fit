import asyncio
from bleak import BleakClient, BleakScanner
import struct
import time

# Put your WIT sensor MAC/address here
# From your WITMotion screenshot it looked like this:
DEVICE_ADDRESS = "E1:05:3D:86:3A:03"

# Common WIT BLE notify characteristic UUIDs
POSSIBLE_NOTIFY_UUIDS = [
    "0000ffe4-0000-1000-8000-00805f9b34fb",
    "0000ffe1-0000-1000-8000-00805f9b34fb",
    "0000fff1-0000-1000-8000-00805f9b34fb",
    "6e400003-b5a3-f393-e0a9-e50e24dcca9e",
]

latest = {
    "ax": 0, "ay": 0, "az": 0,
    "wx": 0, "wy": 0, "wz": 0,
    "angle_x": 0, "angle_y": 0, "angle_z": 0,
}


def signed_int16(low, high):
    value = (high << 8) | low
    if value >= 32768:
        value -= 65536
    return value


def parse_wit_frame(frame):
    """
    WIT frame format usually:
    55 51 ... acceleration
    55 52 ... gyro
    55 53 ... angle
    """

    if len(frame) < 11:
        return

    if frame[0] != 0x55:
        return

    frame_type = frame[1]

    v0 = signed_int16(frame[2], frame[3])
    v1 = signed_int16(frame[4], frame[5])
    v2 = signed_int16(frame[6], frame[7])

    if frame_type == 0x51:
        latest["ax"] = round(v0 / 32768 * 16, 4)
        latest["ay"] = round(v1 / 32768 * 16, 4)
        latest["az"] = round(v2 / 32768 * 16, 4)

    elif frame_type == 0x52:
        latest["wx"] = round(v0 / 32768 * 2000, 4)
        latest["wy"] = round(v1 / 32768 * 2000, 4)
        latest["wz"] = round(v2 / 32768 * 2000, 4)

    elif frame_type == 0x53:
        latest["angle_x"] = round(v0 / 32768 * 180, 4)
        latest["angle_y"] = round(v1 / 32768 * 180, 4)
        latest["angle_z"] = round(v2 / 32768 * 180, 4)


def notification_handler(sender, data):
    data = bytearray(data)

    # A notification can contain multiple 11-byte frames
    for i in range(0, len(data) - 10):
        if data[i] == 0x55:
            frame = data[i:i + 11]
            parse_wit_frame(frame)

    print(
        f"ax={latest['ax']}, ay={latest['ay']}, az={latest['az']} | "
        f"wx={latest['wx']}, wy={latest['wy']}, wz={latest['wz']} | "
        f"angle_x={latest['angle_x']}, angle_y={latest['angle_y']}, angle_z={latest['angle_z']}"
    )


async def find_notify_characteristic(client):
    services = client.services

    print("\nAvailable services and characteristics:")
    for service in services:
        print(f"Service: {service.uuid}")
        for char in service.characteristics:
            print(f"  Characteristic: {char.uuid} | Properties: {char.properties}")

    for uuid in POSSIBLE_NOTIFY_UUIDS:
        try:
            char = client.services.get_characteristic(uuid)
            if char:
                return uuid
        except:
            pass

    # fallback: choose first notify characteristic
    for service in services:
        for char in service.characteristics:
            if "notify" in char.properties or "indicate" in char.properties:
                return char.uuid

    return None


async def main():
    print("Searching for BLE devices...")
    devices = await BleakScanner.discover(timeout=5)

    print("\nFound devices:")
    for d in devices:
        print(f"{d.name} | {d.address}")

    print(f"\nConnecting to: {DEVICE_ADDRESS}")

    async with BleakClient(DEVICE_ADDRESS) as client:
        print("Connected:", client.is_connected)

        notify_uuid = await find_notify_characteristic(client)

        if notify_uuid is None:
            print("No notify characteristic found.")
            return

        print("Using notify characteristic:", notify_uuid)

        await client.start_notify(notify_uuid, notification_handler)

        print("\nReading live WIT sensor data. Press CTRL + C to stop.\n")

        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())