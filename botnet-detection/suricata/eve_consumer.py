import os
from pathlib import Path

class EveConsumer:
    def __init__(self, path: Path, from_end=False):
        self.path = path
        self.offset = os.path.getsize(path) if from_end else 0
        self.restarted = False

    def read_new_lines(self, max_lines: int) -> list[str]:
        size = os.path.getsize(self.path)
        self.restarted = size < self.offset
        if self.restarted:
            print(f"{self.path} was truncated, reading it from the start")
            self.offset = 0

        lines = []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                self.offset += len(raw)
                lines.append(raw.decode("utf-8", errors="replace"))
                if len(lines) >= max_lines:
                    break
        return lines