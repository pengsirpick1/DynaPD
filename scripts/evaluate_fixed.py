"""DMMPv3 fixed DF/RF attacker evaluation entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _has_flag(args: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(flag + "=") for item in args)


def _with_default(args: list[str], flag: str, value: str) -> list[str]:
    if _has_flag(args, flag):
        return args
    return [*args, flag, value]


def main() -> None:
    args = sys.argv[1:]
    if "--dry-run" in args:
        print("DMMPv3 fixed evaluation delegates to scripts/run_attack_eval.py with fixed_df,fixed_rf.")
        return
    args = _with_default(args, "--attackers", "fixed_df,fixed_rf")
    args = _with_default(args, "--adaptive_protocol", "fixed")
    sys.argv = [sys.argv[0], *args]
    from run_attack_eval import main as run_attack_eval_main

    run_attack_eval_main()


if __name__ == "__main__":
    main()
