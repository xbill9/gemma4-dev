"""`litert-lm serve` with constrained decoding off.

litert-lm 0.19.0's OpenAI handler builds ConstrainedDecodingConfig(enable=True) for every chat request,
so any request with `tools` decodes under the engine's tool-call constraint. This wrapper replaces the
class the handler builds with one whose `enable` is False, then runs the stock CLI. Nothing else changes.
"""

import dataclasses
import sys

import litert_lm
from litert_lm_cli.main import main


@dataclasses.dataclass
class _Off(litert_lm.ConstrainedDecodingConfig):
    def __post_init__(self):
        self.enable = False


litert_lm.ConstrainedDecodingConfig = _Off

if __name__ == "__main__":
    sys.argv[0] = "litert-lm"
    sys.exit(main())
