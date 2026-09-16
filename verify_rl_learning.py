"""Predeclared six-episode RL training and six paired held-out evaluations.
Runs simulation processes only; no agents, LLM calls or live learning writes.
"""
import concurrent.futures as cf
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/rl_learning_verification_final_20260915"
SEEDS = [101, 202, 303, 404, 505, 606]
OUT.mkdir(parents=True, exist_ok=True)

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def hashes():
    paths = sorted(set(ROOT.glob("src/**/*.py")) | set(ROOT.glob("config/**/*.json")) | {ROOT / "rl_learning_experiment.py"})
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

protocol_path = OUT / "protocol.json"
INITIAL = (json.loads(protocol_path.read_text(encoding="utf-8"))["source_hashes"]
           if protocol_path.exists() else hashes())
if hashes() != INITIAL:
    raise RuntimeError("Current source differs from the predeclared experiment; refusing mixed-code results")
manifest = {"training_seed":42,"training_episodes":6,"held_out_seeds":SEEDS,
            "ticks_per_episode":51840,"carry_strategic_and_tactical":True,
            "evaluation":"learning and exploration disabled; final checkpoint only, no best-run selection",
            "primary_metric":"paired mean final colony score difference",
            "secondary_metrics":["day330 score","survivors","processing errors"],
            "positive_evidence_rule":"positive mean score difference with two-sided exact paired sign-flip p <= 0.05, no lower mean survivors, no processing errors",
            "small_sample_limitation":"Six predeclared pairs; report exact paired test and all outcomes, including failures.",
            "source_hashes":INITIAL}
if not protocol_path.exists():
    save("protocol.json",manifest)

def run(label, extra):
    if hashes() != INITIAL:
        raise RuntimeError("Source changed during fixed-code experiment")
    destination = OUT / label
    episode = ("train_6" if label == "training" else
               "fresh_frozen_control" if label.startswith("control_") else "trained_frozen_evaluation")
    result_path = destination / episode / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["ticks"] != 51840 or result["processing_errors"]:
            raise RuntimeError("Invalid completed result: " + str(result_path))
        if label != "training":
            checks = json.loads(result_path.with_name("freeze_checks.json").read_text(encoding="utf-8"))
            if not all(checks[key] for key in ("strategic", "operational", "existing_tactical_values")):
                raise RuntimeError("Evaluation was not frozen")
        print("retained " + label, flush=True)
        return
    if label == "training" and destination.exists():
        raise RuntimeError("Partial training requires checkpoint continuation, refusing to reset")
    partial = destination / episode
    if partial.exists():
        archive = destination / ("interrupted_" + str(time.time_ns()))
        for path in (partial, archive):
            if not path.resolve().is_relative_to(OUT.resolve()):
                raise RuntimeError("Archive path outside experiment")
        partial.rename(archive)
    with (OUT / (label + ".log")).open("a", encoding="utf-8") as log:
        log.write("\nResumed evaluation at " + str(time.time()) + "\n")
        log.flush()
        subprocess.run([sys.executable,"-B","rl_learning_experiment.py","--output",str(destination),
                        "--ticks","51840",*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print("completed " + label,flush=True)

def controls():
    for seed in SEEDS:
        run("control_"+str(seed),["--control","--seed",str(seed),"--carry-tactical"])

try:
    save("status.json",{"phase":"training_and_controls","started":time.time()})
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,"training",["--episodes","6","--seed","42","--carry-tactical","--skip-evaluation"]),pool.submit(controls)]
        for job in cf.as_completed(jobs): job.result()
    checkpoint=OUT / "training/train_6"
    save("status.json",{"phase":"held_out_trained_evaluations"})
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,"trained_"+str(seed),["--evaluate-from",str(checkpoint),"--seed",str(seed)]) for seed in SEEDS]
        for job in cf.as_completed(jobs): job.result()
    def read(label, episode):
        return json.loads((OUT / label / episode / "result.json").read_text(encoding="utf-8"))
    def brief(d):
        day330=next(row["score"]["overall"] for row in d["checkpoints"] if row["tick"]==47520)
        return {"score":d["score"]["overall"],"day330":day330,"alive":d["alive"],"errors":len(d["processing_errors"]),"q_frozen":d["q_values_unchanged"]}
    pairs=[{"seed":seed,"control":brief(read("control_"+str(seed),"fresh_frozen_control")),"trained":brief(read("trained_"+str(seed),"trained_frozen_evaluation"))} for seed in SEEDS]
    deltas=[row["trained"]["score"]-row["control"]["score"] for row in pairs]
    mean=statistics.mean(deltas)
    permutations=[statistics.mean(d*s for d,s in zip(deltas,signs)) for signs in itertools.product([-1,1],repeat=len(deltas))]
    p=sum(abs(value)>=abs(mean)-1e-9 for value in permutations)/len(permutations)
    training=[brief(read("training","train_"+str(i))) for i in range(1,7)]
    result={"training":training,"pairs":pairs,"mean_score_gain":mean,"paired_score_deltas":deltas,
            "exact_two_sided_sign_flip_p":p,"mean_day330_gain":statistics.mean(row["trained"]["day330"]-row["control"]["day330"] for row in pairs),
            "mean_survivor_gain":statistics.mean(row["trained"]["alive"]-row["control"]["alive"] for row in pairs),
            "source_hashes_unchanged":hashes()==INITIAL,
            "conclusion":("Evidence of a positive mean learning effect on the held-out set" if mean>0 and p<=0.05 and all(row["trained"]["alive"]>=row["control"]["alive"] for row in pairs) else "Positive descriptive improvement, not established by the paired significance test" if mean>0 else "No positive mean improvement demonstrated"),
            "limitation":"Six held-out pairs in one mission scenario; this does not establish monotonic learning or universal generalization."}
    save("comparison.json",result)
    report=f"# RL learning verification\n\nTraining scores: {', '.join(str(x['score']) for x in training)}.\n\nHeld-out mean score change: {mean:+.2f} points. Two-sided paired sign-flip p: {p:.4f}.\n\n"+result["conclusion"]+". "+result["limitation"]+"\n\n| Seed | Fresh | Trained | Difference |\n|---|---:|---:|---:|\n"
    for row,delta in zip(pairs,deltas): report+=f"| {row['seed']} | {row['control']['score']} | {row['trained']['score']} | {delta:+.1f} |\n"
    (OUT / "REPORT.md").write_text(report,encoding="utf-8")
    save("status.json",{"phase":"complete","mean_score_gain":mean})
except Exception as exc:
    save("status.json",{"phase":"failed","error":str(exc)})
    raise
