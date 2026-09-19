"""Named, independent random streams derived from one seed.

Each concern (arrivals per approach, drivers, detector noise, pedestrians) draws from its own
stream. Changing how often one component samples never shifts the numbers another component
sees. This is what makes controller comparisons use *common random numbers*: every controller
faces exactly the same vehicles, drivers and pedestrians.
"""

from __future__ import annotations

import hashlib

import numpy as np


class RandomStreams:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
            key = int.from_bytes(digest, "little")
            self._streams[name] = np.random.default_rng(np.random.SeedSequence([self.seed, key]))
        return self._streams[name]
