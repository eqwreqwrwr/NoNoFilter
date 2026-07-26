from __future__ import annotations
import sys
import os
from PySide6.QtCore import QLoggingCategory

# Silence only the most noisy technical warnings from Qt Multimedia
QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg.warning=false")

# Workaround for Python 3.12+ compatibility with legacy 'six' importers
# This adds the missing '_path' attribute that Python 3.12's inspection system expects.
for importer in sys.meta_path:
    if "SixMetaPathImporter" in str(type(importer)):
        if not hasattr(importer, "_path"):
            importer._path = []

# Emergency imports for Python 3.12+ compatibility
try:
    import six
    import six.moves
except ImportError:
    pass

# Emergency import for compatibility with NumPy 1.x and 2.x legacy aliases
try:
    import warnings
    import numpy as np
    # Silence technical FutureWarnings from the bridge logic
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # Restore legacy aliases removed in NumPy 1.24+
        if not hasattr(np, 'long'): np.long = int
        if not hasattr(np, 'float'): np.float = float
        if not hasattr(np, 'int'): np.int = int
        if not hasattr(np, 'bool'): np.bool = bool
        if not hasattr(np, 'object'): np.object = object
        # Add new aliases introduced in NumPy 2.0 (for compatibility with newer libs on older NumPy)
        if not hasattr(np, 'ulong'): np.ulong = getattr(np, 'uint', getattr(np, 'uint64', None))

    # Fix numpy.exceptions mismatch (libs expecting NumPy 2.0 exceptions folder on 1.x)
    try:
        import numpy.exceptions as np_exc
        if not hasattr(np_exc, 'RankWarning'): np_exc.RankWarning = np.RankWarning
    except ImportError:
        # If the folder doesn't exist at all, create a fake one for the AI to find
        class FakeExc: pass
        fe = FakeExc()
        fe.RankWarning = np.RankWarning
        sys.modules['numpy.exceptions'] = fe
except ImportError:
    pass

import argparse
from pathlib import Path
from .asr import transcribe
from .audio import render
from .gui import launch
from .matching import find_matches
from .models import load_flags, load_terms, load_transcript, save_flags

def main():
    parser = argparse.ArgumentParser(description="Local audiobook profanity review and censoring")
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui")
    t = sub.add_parser("transcribe"); t.add_argument("audio"); t.add_argument("--out", required=True); t.add_argument("--model", default="small"); t.add_argument("--resume", action="store_true")
    m = sub.add_parser("match"); m.add_argument("transcript"); m.add_argument("terms"); m.add_argument("--out", required=True)
    r = sub.add_parser("render"); r.add_argument("audio"); r.add_argument("flags"); r.add_argument("--out", required=True); r.add_argument("--padding", type=float, default=.10)
    args = parser.parse_args()
    if args.command == "gui": return launch()
    if args.command == "transcribe":
        existing_words, existing_rows = (None, None)
        start_from = 0
        if args.resume and Path(args.out).exists():
            existing_words, existing_rows = load_transcript(args.out)
            if existing_rows:
                start_from = existing_rows[-1]['end']
        transcribe(args.audio, args.out, args.model, start_from=start_from, existing_words=existing_words, existing_rows=existing_rows)
        return
    if args.command == "match": words, _ = load_transcript(args.transcript); save_flags(args.out, find_matches(words, load_terms(args.terms))); return
    if args.command == "render": render(args.audio, load_flags(args.flags), args.out, args.padding)

if __name__ == "__main__": main()
