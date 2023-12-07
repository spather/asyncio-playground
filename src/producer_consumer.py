"""Coding challenge generated by ChatGPT:
Problem Statement:

* Implement a producer-consumer scenario using asyncio where the producer
  generates random numbers at random intervals, and the consumer processes
  these numbers (e.g., prints them or writes to a file).
* The producer and consumer should run concurrently. Use an asyncio Queue to
  facilitate communication between them.
* Ensure that the consumer stops processing after consuming a specific
  number of items or after a timeout.

Skills Tested: Advanced asyncio usage, understanding of producer-consumer
               pattern, working with asyncio Queues, handling timeouts and
               stopping conditions.
"""

import asyncio
import random
import time


async def producer(queue: asyncio.Queue):
    try:
        while True:
            item = random.randint(0, 10)
            queue.put_nowait(item)
            sleep_time = random.uniform(0.01, 2.0)
            print(f"PRODUCER: enqueued {item}, will sleep for {sleep_time:.2f}")
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print(f"PRODUCER: Received CancelledError")
        raise


async def consumer(queue: asyncio.Queue, max_items_to_consume: int, timeout: int):
    print(
        f"CONSUMER: started with max_items_to_consume={max_items_to_consume}, and timeout={timeout}"
    )
    items_consumed = 0
    start_time = time.monotonic()

    while (
        items_consumed < max_items_to_consume
        and time.monotonic() - start_time < timeout
    ):
        item = await queue.get()
        items_consumed += 1
        print(
            f"CONSUMER: received {item}, has been running for {time.monotonic() - start_time:.2f} seconds"
        )

    print(
        f"CONSUMER finished after consuming {items_consumed} items and {time.monotonic() - start_time:.2f} seconds"
    )


async def main():
    queue = asyncio.Queue()

    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(
        consumer(queue, max_items_to_consume=10, timeout=9)
    )

    await consumer_task
    print(f"Consumer task finished, canceling producer task")

    producer_task.cancel()
    try:
        await producer_task
    except asyncio.CancelledError:
        assert producer_task.cancelled
        print(f"Producer task is canceled")

    print("Finished")


if __name__ == "__main__":
    asyncio.run(main())
