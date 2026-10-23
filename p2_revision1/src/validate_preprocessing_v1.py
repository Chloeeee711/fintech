"""Independent verification of real financial output, fingerprints and baseline."""
import ast
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from preprocessing_v1 import FORMAL, sha256


def main():
    latest = json.loads((FORMAL / 'p2_revision1' / 'latest_run.json').read_text())
    run = Path(latest['run_dir'])
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
    status = json.loads((run / 'status.json').read_text())
    provenance = json.loads((run / 'provenance.json').read_text())
    check('input fingerprints match', all(sha256(f) == h for f, h in provenance['input_sha256'].items()))
    check('code fingerprints match', all(sha256(f) == h for f, h in provenance['source_code_sha256'].items()))
    expected = json.loads((run / 'output_sha256.json').read_text())
    check('base output fingerprints match', all(sha256(run / f) == h for f, h in expected.items()))
    source = Path(status['source_run'])
    raw = pd.read_parquet(source / 'data' / 'annual_scope.parquet').set_index(['gvkey', 'fyear'])
    out = pd.read_parquet(run / 'data' / 'financial_panel.parquet')
    check('no financial row loss or duplication', len(raw) == len(out) and not out.duplicated(['gvkey', 'fyear']).any())
    for target in ['d_capex_at', 'droa']:
        errors = []
        for row in out[out[target].notna()].sample(n=min(500, out[target].notna().sum()), random_state=42).itertuples():
            a = raw.loc[(row.gvkey, row.fyear - 1)]
            b = raw.loc[(row.gvkey, row.fyear)]
            c = raw.loc[(row.gvkey, row.fyear + 1)]
            truth = (c['capx'] / b['at'] - b['capx'] / a['at']) if target == 'd_capex_at' else (c['ni'] / ((b['at'] + c['at']) / 2) - b['ni'] / ((a['at'] + b['at']) / 2))
            errors.append(np.isclose(getattr(row, target), truth, rtol=1e-12, atol=1e-12))
        check(target + ' agrees with 500 raw fiscal-key calculations', all(errors) and len(errors) > 0)
        check(target + ' missing Y has missing directional class', out.loc[out[target].isna(), target + '_up'].isna().all())
    audit = pd.read_parquet(run / 'data' / 'filing_match_audit.parquet')
    check('no guessed fiscal matches', status['aligned_rows'] == int(audit.period_match_status.eq('matched').sum()))
    check('real-data feature fitting remains blocked', not status['fit_transform_executed'] and not status['training_ready'])
    working = FORMAL / 'p2_revision1' / 'src'
    baseline = FORMAL / 'vendor' / 'p2_code_original'
    check('RL source retained byte-identical', (working / 'rl_portfolio.py').read_bytes() == (baseline / 'src' / 'rl_portfolio.py').read_bytes())
    check('DL source retained byte-identical', (working / 'transformer_encoder.py').read_bytes() == (baseline / 'src' / 'transformer_encoder.py').read_bytes())
    zip_path = FORMAL.parents[1] / 'p2_code.zip'
    with zipfile.ZipFile(zip_path) as archive:
        check('original imported baseline remains identical to ZIP', all((baseline / item.filename).read_bytes() == archive.read(item) for item in archive.infolist() if not item.is_dir()))
    files = list(working.rglob('*.py'))
    for file in files:
        ast.parse(file.read_text())
    check('all working Python files parse', True)
    result = {'checks': checks, 'passed': sum(c['passed'] for c in checks),
              'total': len(checks), 'python_files': len(files)}
    (run / 'validation.json').write_text(json.dumps(result, indent=2))
    (run / '校验记录.md').write_text('# 第一版修正：真实输出独立校验\n\n'
        + '\n'.join('- ' + ('通过：' if c['passed'] else '失败：') + c['check'] for c in checks)
        + '\n\n这验证财务构造和工程隔离，不代表已核实原文语义或已完成训练。\n')
    print(json.dumps(result, indent=2))
    if result['passed'] != result['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
