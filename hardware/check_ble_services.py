import asyncio
from bleak import BleakClient

ADDRESS = "E1:05:3D:86:3A:03"

async def main():
    print("Connecting to:", ADDRESS)

    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)
        print("\nServices and characteristics:\n")

        for service in client.services:
            print("SERVICE:", service.uuid)

            for char in service.characteristics:
                print("  CHAR:", char.uuid)
                print("  Properties:", char.properties)

asyncio.run(main())
