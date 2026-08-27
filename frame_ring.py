# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Credit-based flow control between a frame producer and a frame consumer.

The producer spends a credit per frame and the consumer returns one per frame, so the ring
cannot overrun by construction rather than by catching a full condition. Watermarks give the
pause hysteresis: the producer stops at `high` and does not resume until the depth falls to
`low`, so a full ring wakes it once instead of once per slot.

Both stalls are counted. Overrun and underflow are not errors, they are the two ways a
pipeline is unbalanced, and a run that hides them reports a rate nobody can act on.

    python frame_ring.py --self-test
"""
from __future__ import annotations

import argparse
import sys
import threading


class Closed(Exception):
    pass


class FrameRing:
    """Fixed slots, one producer, one consumer. A full ring blocks rather than drops."""

    def __init__(self, capacity, name="ring", high=None, low=None):
        if capacity < 1:
            raise ValueError("capacity must be at least one slot")
        self.capacity = int(capacity)
        self.high = int(high if high is not None else self.capacity)
        self.low = int(low if low is not None else max(1, self.high // 2))
        if not 0 < self.low <= self.high <= self.capacity:
            raise ValueError("need 0 < low <= high <= capacity, got %d, %d, %d"
                             % (self.low, self.high, self.capacity))
        self.name = name
        self.credits = self.capacity
        self.paused = False
        self.pauses = 0
        self._slots = [None] * self.capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._closed = False
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self.put_count = 0
        self.get_count = 0
        self.overrun_waits = 0
        self.underflow_waits = 0
        self.high_water = 0

    def put(self, frame):
        with self._not_full:
            if self._count >= self.high and not self.paused:
                self.paused = True
                self.pauses += 1
            while self.paused and self._count > self.low and not self._closed:
                self.overrun_waits += 1
                self._not_full.wait()
            if self._count <= self.low:
                self.paused = False
            if self._closed:
                raise Closed("%s is closed" % self.name)
            if self.credits <= 0:
                raise RuntimeError("%s spent a credit it did not hold" % self.name)
            self.credits -= 1
            self._slots[self._tail] = frame
            self._tail = (self._tail + 1) % self.capacity
            self._count += 1
            self.put_count += 1
            self.high_water = max(self.high_water, self._count)
            self._not_empty.notify()

    def get(self, timeout=None):
        with self._not_empty:
            while self._count == 0:
                if self._closed:
                    raise Closed("%s is drained" % self.name)
                self.underflow_waits += 1
                if not self._not_empty.wait(timeout):
                    raise TimeoutError("%s starved for %.1fs" % (self.name, timeout))
            frame = self._slots[self._head]
            self._slots[self._head] = None
            self._head = (self._head + 1) % self.capacity
            self._count -= 1
            self.get_count += 1
            self.credits += 1
            if self._count <= self.low:
                self.paused = False
                self._not_full.notify()
            return frame

    def close(self):
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def drain(self):
        """Everything still queued, after the producer has closed."""
        out = []
        while True:
            try:
                out.append(self.get(timeout=0.1))
            except (Closed, TimeoutError):
                return out

    def stats(self):
        return {"capacity": self.capacity, "high": self.high, "low": self.low,
                "put": self.put_count, "get": self.get_count, "credits": self.credits,
                "overrun_waits": self.overrun_waits, "underflow_waits": self.underflow_waits,
                "pauses": self.pauses, "high_water": self.high_water, "in_flight": self._count}

    def report(self):
        s = self.stats()
        return ("%s  %d in %d out  high water %d of %d (pause %d, resume %d)  "
                "pauses %d  overrun waits %d  underflow waits %d  credits %d"
                % (self.name, s["put"], s["get"], s["high_water"], s["capacity"],
                   s["high"], s["low"], s["pauses"], s["overrun_waits"],
                   s["underflow_waits"], s["credits"]))


def pump(ring, produce, consume, count):
    """Run a producer thread against a consumer on this thread. Returns the ring."""
    error = []

    def producer():
        try:
            for i in range(count):
                ring.put(produce(i))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            ring.close()

    thread = threading.Thread(target=producer, name="produce", daemon=True)
    thread.start()
    got = 0
    while got < count:
        try:
            consume(ring.get(timeout=600))
        except Closed:
            break
        got += 1
    thread.join()
    if error:
        raise error[0]
    if got != count:
        raise RuntimeError("%s delivered %d of %d frames" % (ring.name, got, count))
    return ring


def self_test():
    """Fifteen controls, over ordering, credits, watermarks and both stalls."""
    import time
    r = []

    ring = FrameRing(4, "order")
    out = []
    pump(ring, lambda i: i, out.append, 50)
    r.append(("every frame arrives, in order", out == list(range(50))))
    r.append(("nothing is put twice", ring.stats()["put"] == 50))
    r.append(("the ring never exceeded its capacity", ring.high_water <= 4))

    slow = FrameRing(2, "overrun")
    pump(slow, lambda i: i, lambda f: time.sleep(0.004), 40)
    r.append(("a slow consumer records overrun waits", slow.overrun_waits > 0))
    r.append(("a slow consumer never exceeds capacity", slow.high_water <= 2))

    starve = FrameRing(16, "underflow")

    def slow_produce(i):
        time.sleep(0.004)
        return i

    pump(starve, slow_produce, lambda f: None, 30)
    r.append(("a slow producer records underflow waits", starve.underflow_waits > 0))
    r.append(("a slow producer keeps the ring nearly empty", starve.high_water <= 3))

    r.append(("the two stalls are told apart",
              slow.overrun_waits > slow.underflow_waits
              and starve.underflow_waits > starve.overrun_waits))

    r.append(("credits are conserved across a run", ring.credits == ring.capacity))
    r.append(("credits never went negative", ring.credits >= 0))

    hyst = FrameRing(8, "hysteresis", high=8, low=2)
    pump(hyst, lambda i: i, lambda f: time.sleep(0.002), 60)
    r.append(("a saturated ring pauses far fewer times than it has frames",
              0 < hyst.pauses < 60))
    r.append(("the watermarks are what bound the depth", hyst.high_water <= 8))

    try:
        FrameRing(4, "bad", high=2, low=3)
        r.append(("low above high is refused", False))
    except ValueError:
        r.append(("low above high is refused", True))

    closed = FrameRing(2, "closed")
    closed.close()
    try:
        closed.put(1)
        r.append(("a closed ring refuses a put", False))
    except Closed:
        r.append(("a closed ring refuses a put", True))

    try:
        FrameRing(0)
        r.append(("a zero-slot ring is refused", False))
    except ValueError:
        r.append(("a zero-slot ring is refused", True))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    print("  " + slow.report())
    print("  " + starve.report())
    print("  " + hyst.report())
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    ap.error("pass --self-test")


if __name__ == "__main__":
    sys.exit(main())
