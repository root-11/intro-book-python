# /// script
# requires-python = ">=3.11"
# ///
"""
crash_consistency.py - the specimen behind §46. A write-ahead log with a per-batch commit marker
(a checksum) recovers, after a torn write, to the last COMMITTED world - never a half-written one.
And the premature-acknowledgement bug: acknowledge before the marker is durable and a crash turns
the acknowledgement into a lie.

    uv run code/measurement/crash_consistency.py

The "crash" is a truncated tail (§46 exercise 1's method): a batch whose marker never landed.
Recovery scans batches, verifies each checksum, and discards the first torn/incomplete one.
Pure standard library - struct, zlib, os - the same machinery sqlite3's WAL mode hardens.
"""
import os
import struct
import tempfile
import zlib

MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def apply_event(world: int, ev: int) -> int:
    # an FNV-style rolling hash standing in for real state: two worlds are equal iff the same
    # committed events were applied in the same order.
    return ((world ^ ev) * 0x0100_0000_01B3) & MASK64


def batch_bytes(evs: list[int]) -> bytes:
    # one batch: [n: u32][events: u64 * n][crc32 over those bytes: u32]. The crc is the commit
    # marker - present and correct means the batch fully landed.
    body = struct.pack("<I", len(evs)) + b"".join(struct.pack("<Q", e) for e in evs)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFF_FFFF)


def recover(path: str) -> tuple[int, int]:
    # apply every batch whose marker verifies; stop at the first torn or incomplete one.
    with open(path, "rb") as f:
        data = f.read()
    pos, world, n = 0, 0, 0
    while True:
        if pos + 4 > len(data):
            break  # no room for a header: clean end or torn
        nev = struct.unpack_from("<I", data, pos)[0]
        body_len = 4 + nev * 8
        if pos + body_len + 4 > len(data):
            break  # batch body or marker truncated: torn tail
        body = data[pos : pos + body_len]
        crc = struct.unpack_from("<I", data, pos + body_len)[0]
        if (zlib.crc32(body) & 0xFFFF_FFFF) != crc:
            break  # marker fails: torn tail
        for k in range(nev):
            (ev,) = struct.unpack_from("<Q", data, pos + 4 + k * 8)
            world = apply_event(world, ev)
        pos += body_len + 4
        n += 1
    return world, n


class Lcg:
    def __init__(self, seed: int) -> None:
        self.s = seed

    def next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.s


def main() -> None:
    path = os.path.join(tempfile.gettempdir(), "crash_consistency.log")

    # --- Scenario 1: torn tail recovers to the last committed world ---
    rng = Lcg(1)
    committed_world = 0
    with open(path, "wb") as f:
        for _ in range(100):
            evs = [rng.next() for _ in range(1 + rng.next() % 8)]
            f.write(batch_bytes(evs))
            f.flush()
            os.fsync(f.fileno())  # fsync barrier: this batch is now durable
            for e in evs:
                committed_world = apply_event(committed_world, e)
        # the crash: a 101st batch begins but the marker never lands - partial body, no crc.
        evs = [11, 22, 33, 44, 55]
        torn = struct.pack("<I", len(evs)) + b"".join(struct.pack("<Q", e) for e in evs[:3])
        f.write(torn)
        f.flush()
        os.fsync(f.fileno())

    world, n = recover(path)
    print("§46 specimen - crash consistency\n")
    print("Scenario 1: torn tail")
    print("  wrote 100 committed batches + 1 torn (partial, no marker)")
    print(f"  recovered {n} batches; world == last committed world: {world == committed_world}")
    assert n == 100, "must discard the torn batch"
    assert world == committed_world, "must recover the last committed world, not a torn one"

    # --- Scenario 2: the premature-acknowledgement lie ---
    def ack_demo(ack_before_marker: bool) -> tuple[int, int]:
        acked = 0
        with open(path, "wb") as f:
            for i in range(50):
                b = batch_bytes([i])
                if ack_before_marker:
                    acked += 1  # told the sender "ok" before fsync
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                else:
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                    acked += 1  # told the sender "ok" only after the marker is durable
            # batch 51 crashes between append and marker.
            if ack_before_marker:
                acked += 1  # acknowledged a batch whose marker will never land
            torn = struct.pack("<I", 1) + struct.pack("<Q", 50)  # body, no crc marker
            f.write(torn)
            f.flush()
            os.fsync(f.fileno())
        _, recovered = recover(path)
        return acked, recovered

    acked_b, rec_b = ack_demo(True)
    acked_a, rec_a = ack_demo(False)
    print("\nScenario 2: premature acknowledgement (crash mid-51st batch)")
    print(f"  ack BEFORE marker: sender holds {acked_b} acks, log recovered {rec_b}  -> {acked_b - rec_b} ack(s) are a lie")
    print(f"  ack AFTER  marker: sender holds {acked_a} acks, log recovered {rec_a}  -> sender and log agree: {acked_a == rec_a}")
    assert acked_b > rec_b, "ack-before-marker must be able to over-acknowledge"
    assert acked_a == rec_a, "ack-after-marker never acknowledges a record the log lost"

    os.remove(path)
    print("\nLogged means one thing: I can read it back after a crash. Everything before the marker is hope.")


if __name__ == "__main__":
    main()
