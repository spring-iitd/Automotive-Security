
import re
import glob
import csv

EPSILON = 0.000025

def parse_log_line(line):
    match = re.match(r"\(([\d\.]+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[(\d+)\]\s+((?:[0-9A-Fa-f]{2} ?)+)", line.strip())
    if not match:
        return None
    timestamp = float(match.group(1))
    can_id = match.group(2).lower().zfill(4)
    dlc = int(match.group(3))
    data_bytes = [byte.lower() for byte in match.group(4).strip().split()]
    data_bytes += ['00'] * (8 - len(data_bytes))
    label = 'T' if set(can_id) <= {'0'} else 'R'
    return [f"{timestamp:.6f}", can_id, str(dlc)] + data_bytes[:8] + [label]

def shift_log_lines(lines, offset):
    shifted = []
    for line in lines:
        entry = parse_log_line(line)
        if entry:
            ts = float(entry[0]) + offset
            entry[0] = f"{ts:.6f}"
            shifted.append(entry)
    return shifted

def get_last_timestamp(entries):
    return max((float(e[0]) for e in entries), default=0.0)

def merge_logs_to_csv(files, output_file='merged.csv'):
    all_entries = []
    offset = 0.0

    for idx, path in enumerate(files):
        with open(path, 'r') as f:
            lines = f.readlines()
        if idx > 0:
            offset += EPSILON
        shifted_entries = shift_log_lines(lines, offset)
        all_entries.extend(shifted_entries)
        offset = get_last_timestamp(shifted_entries)

    if not all_entries:
        print("No valid entries found.")
        return

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_entries)

    print(f"{len(all_entries)} entries written to '{output_file}' (no header).")

# Example usage
if __name__ == "__main__":
    log_files = sorted(glob.glob("*.log"))  # Adjust if needed
    merge_logs_to_csv(log_files)

