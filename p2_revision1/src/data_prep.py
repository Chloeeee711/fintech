"""Revision 1: build audited financial labels, then exact-match actual filings."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import config
from preprocessing_v1 import FORMAL, financial_panel, import_metadata, align_filings, sha256


def build():
    latest = json.loads((FORMAL / 'latest_run.json').read_text())
    source = FORMAL / latest['run_dir']
    if config.PREPROCESS_RUN:
        run = Path(config.PREPROCESS_RUN).resolve()
        if run.exists():
            raise ValueError('Output run must be new: ' + str(run))
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        run = FORMAL / 'p2_revision1' / 'runs' / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / 'data').mkdir()
    status = {'status': 'started', 'source_run': str(source), 'training_ready': False}
    def persist():
        (run / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))
    persist()
    try:
        inputs = [source / 'data' / n for n in ['annual_scope.parquet', 'text_index_clean.parquet', 'text_manifest.parquet']]
        inputs += [source / 'config.json', Path(config.SEC_INDEX)]
        if Path(config.METADATA_PATH).exists():
            inputs.append(Path(config.METADATA_PATH))
        provenance = {'input_sha256': {str(f): sha256(f) for f in inputs},
                      'source_code_sha256': {str(f): sha256(f) for f in Path(config.SRC).glob('*.py')},
                      'metadata_path': config.METADATA_PATH,
                      'rules': {'main_y': 'capx_next/at - capx/at_prev', 'roa': 'ni/mean(at_prev,at)',
                                'filing_window': [2016, 2023], 'annual_interval_days': [330, 400],
                                'train_label_cutoff': '2020-01-01', 'validation_label_cutoff': '2022-01-01'}}
        (run / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        panel = financial_panel(pd.read_parquet(inputs[0]))
        panel.to_parquet(run / 'data' / 'financial_panel.parquet', index=False)
        # Include off-window filings for next-year label publication dates (e.g. 2024).
        from preprocessing_v1 import normalize_accession, normalize_identifier
        idx = pd.read_csv(config.SEC_INDEX, dtype=str)
        idx['accession_key'] = idx.accession_number_nodash.map(normalize_accession)
        idx['gvkey'] = idx.gvkey.map(lambda x: normalize_identifier(x, 6))
        idx['cik'] = idx.cik.map(normalize_identifier)
        idx['filing_year'] = pd.to_numeric(idx.year, errors='coerce').astype('Int64')
        if idx.accession_key.isna().any():
            raise ValueError('Raw index has invalid accessions; review before matching.')
        idx = import_metadata(idx, config.METADATA_PATH)
        aligned, match_audit = align_filings(idx, panel)
        match_audit.to_parquet(run / 'data' / 'filing_match_audit.parquet', index=False)
        if not aligned.empty:
            aligned = aligned[aligned.file_year.between(2016, 2023)].copy()
        manifest = pd.read_parquet(source / 'data' / 'text_manifest.parquet')
        if not aligned.empty:
            aligned = aligned.merge(manifest[['accession_key', 'read_status', 'requires_review', 'clean_sha256']],
                                    on='accession_key', how='left', validate='one_to_one')
            # Quarantine ambiguous/short/duplicate texts until reviewed; no empty-text substitution.
            aligned['text_eligible'] = aligned.read_status.eq('ok') & aligned.requires_review.eq(False)
            aligned['main_eligible'] &= aligned.text_eligible
            aligned['secondary_eligible'] &= aligned.text_eligible
            aligned['index'] = range(len(aligned))
            aligned['text_path'] = aligned.file_year.astype(str) + '/' + aligned.filename
        aligned.to_parquet(run / 'data' / 'analytical_sample.parquet', index=False)
        pending = match_audit[match_audit.filing_year.between(2016, 2023) & ~match_audit.metadata_ready]
        pending[['accession_key', 'cik', 'gvkey', 'filing_year', 'filename']].to_csv(run / 'pending_metadata.csv', index=False)
        scope = panel.fyear.between(2015, 2022)
        status.update(status='completed', financial_rows=len(panel),
                      financial_main_y_nonmissing=int(panel.main_y_valid.sum()),
                      financial_secondary_y_nonmissing=int(panel.secondary_y_valid.sum()),
                      financial_2015_2022_rows=int(scope.sum()),
                      financial_2015_2022_main_y_nonmissing=int(panel.loc[scope, 'main_y_valid'].sum()),
                      financial_2015_2022_secondary_y_nonmissing=int(panel.loc[scope, 'secondary_y_valid'].sum()),
                      pending_metadata_rows=len(pending), aligned_rows=len(aligned),
                      fit_transform_executed=False, neural_and_rl_executed=False,
                      training_ready=False,
                      note='Financial candidates are not matched text samples. Review metadata, text quarantine, historical availability and downstream training before enabling models.')
        for task in ['main', 'secondary']:
            if not aligned.empty:
                subset = aligned[aligned[task + '_eligible'] & aligned.split.ne('excluded')]
                subset.to_parquet(run / 'data' / (task + '_sample.parquet'), index=False)
                status[task + '_split_counts'] = subset.split.value_counts().to_dict()
        persist()
        outputs = {str(f.relative_to(run)): sha256(f) for f in run.rglob('*') if f.is_file()}
        (run / 'output_sha256.json').write_text(json.dumps(outputs, indent=2))
        (run / '运行记录.md').write_text('# 第一版预处理运行记录\n\n'
            + '源清洗run：`' + str(source) + '`。原文件不修改。\n\n'
            + '顺序：校验输入指纹 → 逐财年构造财务Y与原始F → 导入真实申报元数据 → 精确匹配 → 检查文本质量 → 时间及标签公开日过滤 → 分任务保存。\n\n'
            + '```json\n' + json.dumps(status, indent=2, ensure_ascii=False) + '\n```\n\n'
            + '尚未拟合标准化、填补或词表。没有运行深度学习、RL或调用Qwen。源代码/输入与输出指纹分别见provenance.json和output_sha256.json。\n')
        (FORMAL / 'p2_revision1' / 'latest_run.json').write_text(json.dumps({'run_dir': str(run), 'status': status}, indent=2))
        print(json.dumps({'run_dir': str(run), **status}, indent=2, ensure_ascii=False))
        return aligned
    except Exception as error:
        status.update(status='failed', error=repr(error))
        persist()
        (run / '运行记录.md').write_text('# 失败运行\n\n' + repr(error) + '\n未生成可训练样本；保留现场。\n')
        raise

if __name__ == '__main__':
    build()
