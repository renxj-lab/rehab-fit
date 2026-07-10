import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for BLE devices for 10 seconds...\n")

    devices = await BleakScanner.discover(timeout=10)

    if not devices:
        print("No BLE devices found.")
        return

    for d in devices:
        print("--------------------------------")
        print("Name:", d.name)
        print("Address:", d.address)
        print("Details:", d.details)

asyncio.run(main())
