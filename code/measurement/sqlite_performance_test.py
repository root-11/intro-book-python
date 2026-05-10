# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import sqlite3
import time
import csv
from pathlib import Path
# Load CSV data manually
cwd = Path(__file__).parent
csv_file = cwd/ "res-dist.csv"
assert csv_file.exists()

data = []
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for i,row in enumerate(reader):
        if i==0:continue # headers
        A = int(row[0])
        B = int(row[1])
        dist = float(row[2])
        data.append((A, B, dist))

# Define a function to insert data and measure lookup throughput
def test_throughput(db_location):
    conn = sqlite3.connect(db_location)
    cursor = conn.cursor()


    cursor.execute("DROP TABLE IF EXISTS distances")
    cursor.execute("""
        CREATE TABLE distances (
            A INTEGER,
            B INTEGER,
            distance_m REAL,
            PRIMARY KEY (A, B)
        ) WITHOUT ROWID
    """)

    conn.execute("BEGIN TRANSACTION")
    cursor.executemany("INSERT INTO distances (A, B, distance_m) VALUES (?, ?, ?)", data)
    conn.commit()

    # Disable journaling and synchronous mode for performance as this is purely READs.
    cursor.execute("PRAGMA journal_mode = OFF")
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA locking_mode = EXCLUSIVE")

    # Prepare lookup test using all inserted pairs
    sample_size = min(100000, len(data))
    lookup_pairs = data[:sample_size]

    total_distance = 0.0
    start_time = time.perf_counter()
    for lp1, lp2, _ in lookup_pairs:
        cursor.execute("SELECT distance_m FROM distances WHERE A = ? AND B = ?", (lp1, lp2))
        total_distance += cursor.fetchone()[0]
    end_time = time.perf_counter()

    conn.close()
    print("checksum: ", total_distance)
    duration = end_time - start_time
    throughput = sample_size / duration
    return throughput

# Run tests
in_memory_throughput = test_throughput(":memory:")
on_disk_throughput = test_throughput(str(cwd / "dist.db"))

print(f"In-memory DB throughput: {in_memory_throughput:,.2f} lookups/sec")
print(f"On-disk DB throughput: {on_disk_throughput:,.2f} lookups/sec")

# OUTPUTS:
# checksum:  6135444.699008807
# checksum:  6135444.699008807
# In-memory DB throughput: 906,488.04 lookups/sec
# On-disk DB throughput: 826,627.98 lookups/sec