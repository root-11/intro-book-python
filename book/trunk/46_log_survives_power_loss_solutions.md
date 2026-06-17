# 46 - The log survives power loss - reference notes

## Exercises 1-2: tear a log, then add the commit marker

A length-prefixed record stream with no marker and no `fsync` decodes whatever bytes survived. Truncate past the last sector and the final record's payload is short: `struct.unpack` reads past the end (an exception), or - worse - reads a plausible-but-wrong value and folds it into the world silently. That silent case is the dangerous one; it is corruption that passes.

The marker fixes it. Append `struct.pack("<I", len(body)) + body + struct.pack("<I", zlib.crc32(body))` and, on replay, recompute the crc over the body and compare. A torn tail fails the check (or the length runs past EOF) and is discarded. The recovered world is the last *committed* world - the boundary between "durable" and "did not occur" is the marker. `crash_consistency.py` is the worked specimen: 100 committed batches plus a torn one recover to exactly the 100.

## Exercise 3: order the barrier

`os.fsync` the records, write the marker, `os.fsync` again. A crash before the first fsync: records may not be durable, but the marker is not either, so replay discards the batch - consistent. Between the two fsyncs: records durable, marker not - replay sees no valid marker, discards the batch - consistent. After the second: both durable - replay accepts - consistent. The one ordering that breaks is writing the marker *before* fsyncing the records: a crash there leaves a valid marker pointing at records that never landed. The marker must be the last thing made durable.

## Exercises 4-6: atomic snapshot, idempotent replay, recover to any tick

`os.replace(tmp, dst)` is atomic, so a crash mid-snapshot leaves the previous snapshot whole; `os.fsync` the directory afterwards so the rename itself survives. Recovery loads the last intact snapshot and replays the committed suffix. It must be idempotent because a crash can land after a batch is durable but before it folds into a snapshot, so the batch may replay twice: always replay from a snapshot that *predates* the batch, and determinism ([§16](16_determinism_by_order.md)) gives the same world every time. Hash the live world at tick T and the recovered world; bit-identical, or the first divergent event is your non-idempotent one.

## Exercise 7: the premature acknowledgement

This is the load-bearing one. Acknowledge *before* the marker is durable and a `kill -9` between append and fsync leaves the sender holding an "ok" for a record the recovered log does not contain - the acknowledgement was a lie. Move the "ok" to *after* the marker and the sender simply retries the un-acknowledged record; the log and the sender agree. `crash_consistency.py`'s second scenario measures it: ack-before-marker over-acknowledges by exactly the torn batch, ack-after never does. "Logged" has one honest definition: *I can read it back after a crash.*

## Exercises 8-9: measure the barrier, price the database

`fsync` per record, per batch, and never: reproduce the [§38](38_storage_systems.md) batching span - the per-record fsync is bound to IOPS, the per-batch one amortises it, and no-fsync is a buffered write that a power loss empties. Then replace the hand-rolled log with `sqlite3` in WAL mode (`conn.execute("PRAGMA journal_mode=WAL")`). It gives you the commit marker, the atomic durability, and the idempotent recovery you just wrote, hardened against edge cases (group commit, partial-page tears, consumer-drive fsync lies) you have not hit - and it costs *nothing* to add, because `sqlite3` is in the standard library. The lesson is not "always hand-roll" or "always SQLite"; it is that you now know which guarantee you are buying and why the bare `file.write()` did not have it.
