"""Read-only static package audit and tiny synthetic formula demonstrations.

Does not import/execute package modules, deserialize models, or train anything.
"""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT/"vendor/p2_code_original"
OUT=PROJECT/"reports/p2_code_review"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    syntax=[]
    imports=set()
    for p in sorted(BASE.rglob("*.py")):
        try:
            tree=ast.parse(p.read_text(),filename=str(p))
            syntax.append({"file":str(p.relative_to(BASE)),"parsed":True})
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):
                    imports.update(a.name.split('.')[0] for a in n.names)
                elif isinstance(n,ast.ImportFrom) and n.module:
                    imports.add(n.module.split('.')[0])
        except SyntaxError as e:
            syntax.append({"file":str(p.relative_to(BASE)),"parsed":False,"error":str(e)})
    # Demonstrate the exact key arithmetic found in both lag construction sites.
    current=pd.DataFrame({"firm":["A"]*3,"fyear":[2018,2019,2020],"score":[10.,20.,30.]})
    prev=current.copy()
    prev["py"]=prev.fyear-1
    lag=current.merge(prev[["firm","py","fyear","score"]],left_on=["firm","fyear"],right_on=["firm","py"],how="left",suffixes=("","_joined"))
    lag=lag.rename(columns={"fyear_joined":"actual_joined_year","score_joined":"claimed_lag_score"})
    lag.to_csv(OUT/"lag_direction_example.csv",index=False)
    # Show that the two target definitions differ even when capex doubles.
    at_prev,at_now,capx_now,capx_next=100.,200.,5.,10.
    target={"inputs":{"at_prev":at_prev,"at_now":at_now,"capx_now":capx_now,"capx_next":capx_next},
            "code_target":(capx_next-capx_now)/at_now,
            "plan_target":capx_next/at_now-capx_now/at_prev}
    # Faithful minimal reproduction of the forward cumulative-max adjustment.
    pvals=[0.01,0.011]
    raw=[p*len(pvals)/(i+1) for i,p in enumerate(pvals)]
    package_adjustment=np.maximum.accumulate(raw).clip(0,1).tolist()
    reverse_min_adjustment=np.minimum.accumulate(raw[::-1])[::-1].clip(0,1).tolist()
    # A valid nonnegative pair of losses can make the package variance negative.
    la=np.array([2.,0.,2.,0.,2.,0.]);lb=np.ones(6)
    d=la-lb
    variance=float(d.var(ddof=1)+2*np.cov(d[:-1],d[1:])[0,1])
    quality=pd.read_csv(BASE/"data/capex_plan_verification_sample.csv")
    rl=json.loads((BASE/"data/rl_results.json").read_text())
    rl_summary={k:{"n_train_years":v["n_train"],"n_total_years":v["n_total"],
                   "exposure_min":min(v["agent"]["exps"]),"exposure_max":max(v["agent"]["exps"]),
                   "reported_agent_sharpe":v["agent"]["sharpe"],"reported_fixed50_sharpe":v["fixed50"]["sharpe"]} for k,v in rl.items()}
    provenance=json.loads((PROJECT/"vendor/p2_code_provenance.json").read_text())
    archive=Path(provenance["archive_path"])
    baseline_matches=True
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not item.is_dir():
                baseline_matches &= (BASE/item.filename).read_bytes()==z.read(item.filename)
    stages=ast.parse((BASE/"src/run_all.py").read_text())
    listed=[]
    for node in stages.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="STAGES" for t in node.targets):
            listed=ast.literal_eval(node.value)
    result={"scope":"Static source review, existing JSON/CSV inspection, own synthetic calculations only; no original modules imported or trained.",
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "original_extraction_byte_identical":bool(baseline_matches),"syntax":syntax,
            "external_model_libraries_imported":sorted(imports&{"transformers","sentence_transformers","openai","dashscope"}),
            "lag_demo":lag.to_dict(orient="records"),"target_demo":target,
            "fdr_demo":{"p":pvals,"package_q":package_adjustment,"correct_reverse_min_q":reverse_min_adjustment},
            "loss_variance_demo":{"loss_a":la.tolist(),"loss_b":lb.tolist(),"package_variance":variance,"negative_variance":variance<0},
            "verification_sample":{"rows":len(quality),"a1_topic_labeled":int(quality.a1_topic.notna().sum()),"a2_topic_labeled":int(quality.a2_topic.notna().sum()),"rule_amount_unique":sorted(quality.rule_amount.dropna().unique().tolist())},
            "run_all_stages":[s[0] for s in listed],"reported_rl_artifacts_not_reproduced":rl_summary}
    (OUT/"evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    assert baseline_matches
    assert all(s["parsed"] for s in syntax)
    assert lag.loc[0,"actual_joined_year"]==2019
    assert target["code_target"]!=target["plan_target"]
    print(json.dumps({"python_files_parsed":len(syntax),"baseline_byte_identical":bool(baseline_matches),"model_library_imports":result["external_model_libraries_imported"],"manual_labels":result["verification_sample"],"formula_demonstrations":"saved in reports/p2_code_review/evidence.json"},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
