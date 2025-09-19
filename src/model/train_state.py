import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainState:
    epoch: int = 0
    loss: float = 0.0

    def save(self, path: Path):
        path.write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "TrainState":
        if path.exists():
            return TrainState(**json.loads(path.read_text(encoding="utf-8")))
        else:
            return TrainState()
