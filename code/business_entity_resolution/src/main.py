"""CLI entry point for the business entity resolution pipeline.

Examples (run from the repository root):
    python code/business_entity_resolution/src/main.py --mode all
    python code/business_entity_resolution/src/main.py --mode predict --tau 0.55
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console safety
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline as P


def main():
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    ap.add_argument("--data-dir", default="utils/datasets",
                    help="dir containing train/ and test/ subfolders")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--model", default="output/model.joblib")
    ap.add_argument("--run-log", default="output/run_log.jsonl")
    ap.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=None,
                    help="decision threshold; if omitted, use value tuned in train")
    ap.add_argument("--train-n", type=int, default=150000)
    ap.add_argument("--val-n", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--no-addr", action="store_true")
    a = ap.parse_args()

    use_pair, use_addr = (not a.no_pair), (not a.no_addr)
    os.makedirs(a.out_dir, exist_ok=True)
    tau, model = a.tau, None

    if a.mode in ("train", "all"):
        result, model, tuned_tau = P.train_and_validate(
            a.data_dir, a.model, a.run_log, seed=a.seed,
            train_n=a.train_n, val_n=a.val_n, K=a.K, cap=a.cap,
            use_pair=use_pair, use_addr=use_addr)
        if tau is None:
            tau = tuned_tau
        with open(os.path.join(a.out_dir, "tau.txt"), "w") as f:
            f.write(str(tau))
        print(json.dumps({k: v for k, v in result.items() if k != "sweep_table"},
                         indent=2, ensure_ascii=False))

    if a.mode in ("predict", "all"):
        import model as M
        if model is None:
            model = M.load(a.model)
        if tau is None:
            tpath = os.path.join(a.out_dir, "tau.txt")
            tau = float(open(tpath).read()) if os.path.exists(tpath) else 0.5
        stats = P.predict_test(a.data_dir, a.out_dir, model, tau, K=a.K, cap=a.cap,
                               use_pair=use_pair, use_addr=use_addr, batch=a.batch)
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
