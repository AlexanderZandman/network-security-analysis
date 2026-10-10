import pandas as pd
from pathlib import Path
from Suricata_Event import Suricata_Event


def parse_eve(file: Path) -> list[Suricata_Event]:
    # read JSON input
    df = pd.read_json(file)
    events = []

    for item in df.itertuples(index=False):
        event = Suricata_Event.from_dict(item._asdict())
        events.append(event)
    
    return events

if __name__ == "__main__":
    file_path = Path("example.json")
    events = parse_eve(file_path)
    print(f"Parsed {len(events)} events")
    for event in events:
        print(event)