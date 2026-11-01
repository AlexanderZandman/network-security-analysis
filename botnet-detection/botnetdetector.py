import json
import pandas as pd
from suricata.events_mapping import map_suricata_events_to_flow_df
from model.predict import load_everything_model, load_models, load_training_reference, predict, predict_everything
from pathlib import Path
from suricata.suricata_event import SuricataEvent
from suricata.suricata_event_logger import SuricataEventLogger

FEATURE_TIME_WINDOW = pd.Timedelta(minutes=1)
MAX_CONTEXT_FLOWS = 250_000
MODEL_CHOICES = ("split", "everything")

class BotnetDetector:
    def __init__(self, threshold=0.5, model="split"):
        if model not in MODEL_CHOICES:
            raise ValueError(f"Unknown model {model!r}, expected one of {MODEL_CHOICES}")
        self.threshold = threshold
        self.model = model
        self.categories = load_training_reference()
        if model == "everything":
            self.everything_classifier, self.everything_classes = load_everything_model()
        else:
            self.botnet_detector, self.attack_classifier, self.attack_classes = load_models()
        self.logger = SuricataEventLogger(Path(__file__).parent / "suricata" / "botnet-detector.log", model)
        self.context_df = None

    def process_flows(self, lines: list[str], fast_log: Path, alert_eve: Path):
        flow_df = map_suricata_events_to_flow_df(_parse_flow_events(lines))
        if flow_df.empty:
            return

        context_df = self._context_for(flow_df)
        if self.model == "everything":
            results = predict_everything(flow_df, self.everything_classifier, self.categories,
                                         self.everything_classes, self.threshold, context_df=context_df)
        else:
            results = predict(flow_df, self.botnet_detector, self.attack_classifier,
                              self.categories, self.attack_classes, self.threshold, context_df=context_df)
        n_alerts = self.logger.write_alerts(results, fast_log, alert_eve, flow_df)
        print(f"Analyzed {len(flow_df)} new flows, raised {n_alerts} alert(s)")
        if n_alerts:
            print(results.loc[results["is_botnet"], "attack_type"].value_counts().to_string())

        self._update_context(flow_df)

    def reset(self):
        self.context_df = None

    def _context_for(self, flow_df):
        if self.context_df is None:
            return None
        buckets = flow_df["ts"].dt.floor(FEATURE_TIME_WINDOW).unique()
        return self.context_df[self.context_df["ts"].dt.floor(FEATURE_TIME_WINDOW).isin(buckets)]

    def _update_context(self, flow_df):
        flows = flow_df if self.context_df is None else pd.concat([self.context_df, flow_df], ignore_index=True)
        if len(flows) > MAX_CONTEXT_FLOWS:
            bucket_sizes = flows["ts"].dt.floor(FEATURE_TIME_WINDOW).value_counts().sort_index(ascending=False)
            kept_buckets = bucket_sizes.index[bucket_sizes.cumsum() <= MAX_CONTEXT_FLOWS]
            flows = flows[flows["ts"].dt.floor(FEATURE_TIME_WINDOW).isin(kept_buckets)]
        self.context_df = flows.reset_index(drop=True)

def _parse_flow_events(lines: list[str]) -> list[SuricataEvent]:
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = SuricataEvent.from_json(line)
        except json.JSONDecodeError:
            print(f"Skipping malformed line: {line[:80]}")
            continue
        if event.is_flow:
            events.append(event)
    return events
