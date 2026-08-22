# ASOC clean decision experiment

This directory is intentionally separate from the legacy pipeline.  It trains one 144-step, time-conditioned GFNODE and one parameter-matched discrete trajectory decoder on the same leakage-free protocol.  Run `python asoc_clean_decision.py --run-all` after installing the listed Python dependencies.

The protocol retains the complete regular five-minute timeline, fits all preprocessors only on Train, constructs windows separately inside each split, and uses Validation only for early stopping.  Test is evaluated once from the validation-best checkpoint.
