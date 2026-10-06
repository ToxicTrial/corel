from __future__ import annotations

import sys

from cutoutnet.trainer import train


def _enable_live_output() -> None:
    # Colab often runs this script as a child process. Force line-buffered output
    # so startup/resume progress is visible immediately instead of appearing
    # only after a large stdout buffer fills.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


if __name__ == "__main__":
    _enable_live_output()
    print("CutoutNet Colab launcher: live output enabled", flush=True)
    train("config/train_colab.yaml")
