from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.target_policy.target_pool import validate_target_policy_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate target_policy_pool_v1 invariants.")
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--write_report", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate_target_policy_pool(args.pool_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.write_report:
        report_path = Path(args.write_report)
        if report_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"Refusing to overwrite existing report: {report_path}. Pass --overwrite to replace it.")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    if args.strict and not bool(report.get("valid")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
