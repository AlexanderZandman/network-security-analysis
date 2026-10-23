from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Flow":
        if not data:
            return cls()
        return cls(
            pkts_toserver=data.get("pkts_toserver", 0),
            pkts_toclient=data.get("pkts_toclient", 0),
            bytes_toserver=data.get("bytes_toserver", 0),
            bytes_toclient=data.get("bytes_toclient", 0),
            start=data.get("start"),
            end=data.get("end"),
            age=data.get("age", 0),
            state=data.get("state"),
            reason=data.get("reason"),
            alerted=data.get("alerted", False),
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
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["TCP"]:
        if not data or (data != data): # make sure event is valid and not NaN
            return None
        return cls(
            tcp_flags=data.get("tcp_flags"),
            tcp_flags_ts=data.get("tcp_flags_ts"),
            tcp_flags_tc=data.get("tcp_flags_tc"),
            syn=data.get("syn", False),
            state=data.get("state"),
            ts_max_regions=data.get("ts_max_regions", 0),
            tc_max_regions=data.get("tc_max_regions", 0),
        )


@dataclass
class Suricata_Event:
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Suricata_Event":
        return cls(
            timestamp=data.get("timestamp"),
            flow_id=data.get("flow_id"),
            event_type=data.get("event_type"),
            src_ip=data.get("src_ip"),
            src_port=data.get("src_port"),
            dest_ip=data.get("dest_ip"),
            dest_port=data.get("dest_port"),
            ip_v=data.get("ip_v"),
            proto=data.get("proto"),
            app_proto=data.get("app_proto"),
            flow=Flow.from_dict(data.get("flow")),
            tcp=TCP.from_dict(data.get("tcp")),
        )