import argparse
from pathlib import Path
from time import sleep
from botnetdetector import BotnetDetector
from suricata.eve_consumer import EveConsumer

DEFAULT_EVE_PATH = Path(__file__).parent / "logs/eve.json"

def main(args):
    print("Starting program...")
    botnet_detector = BotnetDetector(threshold=args.threshold)
    eve_tracker = EveConsumer(args.input, from_end=args.from_end)
    print(f"Following {args.input}, alerts go to {args.fast_log} and {args.alert_eve}")

    while True:
        lines = eve_tracker.read_new_lines(args.batch_size)
        if eve_tracker.restarted:
            # Flows from the previous file must not count towards the new flows' features
            botnet_detector.reset()
        if lines:
            botnet_detector.process_flows(lines, args.fast_log, args.alert_eve)
        if len(lines) < args.batch_size:
            # Caught up with the file; wait for Suricata to log new data
            sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_EVE_PATH,
                        help="Path to Suricata's eve.json to follow (default: botnet-detection/logs/eve.json)")
    parser.add_argument("--from-end", action="store_true",
                        help="Skip the lines already in the file and only analyze lines appended from now on")
    parser.add_argument("-b", "--batch-size", type=int, default=50000,
                        help="Maximum number of lines analyzed at once (default: 50000)")
    parser.add_argument("--interval", type=float, default=3,
                        help="Seconds to wait for new lines once the end of the file is reached (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Botnet probability threshold (default: 0.5)")
    parser.add_argument("--fast-log", type=Path, default=None,
                        help="Path to append fast.log-style alerts (default: <input dir>/ml_fast.log)")
    parser.add_argument("--alert-eve", type=Path, default=None,
                        help="Path to append eve.json-style alerts (default: <input dir>/ml_eve.json)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")
    args.fast_log = args.fast_log or args.input.parent / "ml_fast.log"
    args.alert_eve = args.alert_eve or args.input.parent / "ml_eve.json"
    main(args)
