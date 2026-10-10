from datetime import timedelta
from time import sleep
import os
import argparse
import pandas as pd
from pathlib import Path
from Suricata_Event import Suricata_Event

def parse_eve(file: Path) -> list[Suricata_Event]:
    # read JSON input
    # TODO: optimize this: in its current state, after every filechange, the ENTIRE file is read
    df = pd.read_json(file)

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    latest = df.iloc[-1].timestamp

    # Filter the dataframe up to one minute before the latest timestamp
    one_minute_ago = latest - timedelta(minutes=1)
    df = df[df['timestamp'] >= one_minute_ago]

    print(df.head())
    events = []

    for item in df.itertuples(index=False):
        event = Suricata_Event.from_dict(item._asdict())
        events.append(event)
    
    return events

def run_detection(args):
    # Parse the eve.json
    events = parse_eve(args.input)

    print(f"Parsed {len(events)} events")

    # TODO: Debug output, remove
    for event in events:
         print(f"  {event.src_ip}:{event.src_port} -> {event.dest_ip}:{event.dest_port} ({event.proto})")

    # Map the events from Suricata to our model

    # Call our model to detect anomalies

    # Output report

def main(args):
    print("Starting program...")
    last_modified = os.path.getmtime(args.input)

    # Poll a file every minute.
    # If it was changed, do a new anomaly detection
    while True:
        current_modified = os.path.getmtime(file_path)

        if current_modified != last_modified:
            print("File has changed!")
            last_modified = current_modified
            run_detection(args)
        
        sleep(60) # wait for a minute

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the file to monitor")
    args = parser.parse_args()

    file_path = args.input

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    main(args)
