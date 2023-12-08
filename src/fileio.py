"""Example of using asyncio for file I/O

This program behaves like the unix `tail` utility. Given a filename,
it will open the file, seek to the end, and then print any new lines
that are added to the end.
"""
import aiofiles
import asyncio
import sys
from typing import Any, Coroutine


async def tail_file(filename: str, target: Coroutine[Any, Any, None]):
    async with aiofiles.open(filename, mode="r") as f:
        await f.seek(0, 2)  # Go to the end of the file
        while True:
            # f.readline() always returns right away with an empty line.
            # I'd prefer if it instead blocked until a line was available.
            line = await f.readline()

            if line is not None and len(line) > 0:
                target.send(line.rstrip("\n"))


def printer():
    while True:
        line = yield
        print(line)


async def main():
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} <filename>")
        return

    filename = sys.argv[1]
    target = printer()
    next(target)  # Prime it

    await tail_file(filename, target)


if __name__ == "__main__":
    asyncio.run(main())
