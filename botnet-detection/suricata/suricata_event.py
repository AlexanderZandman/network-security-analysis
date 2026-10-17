import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _as_dict(data: Any) -> Optional[Dict[str, Any]]:
    return data if isinstance(data, dict) and data else None

# Nan values in Suricata EVE JSON are represented as NaN, which is not valid JSON.
def _check_nan(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def _as_int(value: Any, default: int = 0) -> int:
    value = _check_nan(value)
    return default if value is None else int(value)


@dataclass
class Flow:
    pkts_toserver: int = 0
    pkts_toclient: int = 0
    bytes_toserver: int = 0
    bytes_toclient: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    age: int = 0
    state: Optional[str] = None
    reason: Optional[str] = None
    alerted: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "Flow":
        data = _as_dict(data)
        if data is None:
            return cls()
        return cls(
            pkts_toserver=_as_int(data.get("pkts_toserver")),
            pkts_toclient=_as_int(data.get("pkts_toclient")),
            bytes_toserver=_as_int(data.get("bytes_toserver")),
            bytes_toclient=_as_int(data.get("bytes_toclient")),
            start=_check_nan(data.get("start")),
            end=_check_nan(data.get("end")),
            age=_as_int(data.get("age")),
            state=_check_nan(data.get("state")),
            reason=_check_nan(data.get("reason")),
            alerted=bool(data.get("alerted", False)),
        )


@dataclass
class TCP:
    tcp_flags: Optional[str] = None
    tcp_flags_ts: Optional[str] = None
    tcp_flags_tc: Optional[str] = None
    syn: bool = False
    state: Optional[str] = None
    ts_max_regions: int = 0
    tc_max_regions: int = 0

    @classmethod
    def from_dict(cls, data: Any) -> Optional["TCP"]:
        data = _as_dict(data)
        if data is None:
            return None
        return cls(
            tcp_flags=data.get("tcp_flags"),
            tcp_flags_ts=data.get("tcp_flags_ts"),
            tcp_flags_tc=data.get("tcp_flags_tc"),
            syn=bool(data.get("syn", False)),
            state=data.get("state"),
            ts_max_regions=_as_int(data.get("ts_max_regions")),
            tc_max_regions=_as_int(data.get("tc_max_regions")),
        )


@dataclass
class SuricataEvent:
    timestamp: Optional[str] = None
    flow_id: Optional[int] = None
    event_type: Optional[str] = None
    src_ip: Optional[str] = None
    src_port: Optional[int] = None
    dest_ip: Optional[str] = None
    dest_port: Optional[int] = None
    ip_v: Optional[int] = None
    proto: Optional[str] = None
    app_proto: Optional[str] = None
    flow: Flow = field(default_factory=Flow)
    tcp: Optional[TCP] = None

    @property
    def is_flow(self) -> bool:
        return self.event_type == "flow"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuricataEvent":
        src_port = _check_nan(data.get("src_port"))
        dest_port = _check_nan(data.get("dest_port"))
        return cls(
            timestamp=_check_nan(data.get("timestamp")),
            flow_id=_check_nan(data.get("flow_id")),
            event_type=_check_nan(data.get("event_type")),
            src_ip=_check_nan(data.get("src_ip")),
            src_port=None if src_port is None else int(src_port),
            dest_ip=_check_nan(data.get("dest_ip")),
            dest_port=None if dest_port is None else int(dest_port),
            ip_v=_check_nan(data.get("ip_v")),
            proto=_check_nan(data.get("proto")),
            app_proto=_check_nan(data.get("app_proto")),
            flow=Flow.from_dict(data.get("flow")),
            tcp=TCP.from_dict(data.get("tcp")),
        )

    @classmethod
    def from_json(cls, line: str) -> "SuricataEvent":
        return cls.from_dict(json.loads(line))
