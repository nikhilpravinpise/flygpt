"""``python -m flygpt.evals`` - run the T-* evaluation battery.

    python -m flygpt.evals --list
    python -m flygpt.evals --results-dir results [--graph-dir data/graph]
        [--corpus-dir data/corpus] [--vocab-dir data/vocab]
        [--banc-dir data/graph_banc] [--banc-graph-dir data/graph_banc]
        [--only T-SIM-1,T-CORPUS-2] [--skip T-REAL]
    python -m flygpt.evals claims --repo-root .
"""

from flygpt.evals import main

raise SystemExit(main())
