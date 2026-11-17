"""Patient-balanced baseline scoring for human visits and saved mouse trajectories."""

def run(inputs, output):
    import numpy as np
    import pandas as pd
    from scipy.io import mmread
    from common.paths import REPO_ROOT as ROOT, HCOP_PATH
    SC_DIR = inputs.results / 'GSE272993_longitudinal/data/exported'
    OUT_DIR = output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH = ROOT / 'data/mouse/program_weights' / 'Fig1e_program_top_genes.csv'
    MOUSE_PATH = inputs.mouse_trajectories
    ALPHA_PATH = inputs.dynamics / 'robustness/dynamic_importance_by_delta.csv'
    from common.validation_utils import PROGRAM_ORDER, SOURCE_TO_MODEL, prepare_weights, load_alpha
    TP_ORDER = ['Baseline', 'Follow Up 1', 'Follow Up 2', 'Follow Up 3', 'Follow Up 4']
    alpha = load_alpha(ALPHA_PATH, treatment='Anti-PD1')
    genes = pd.Index((SC_DIR / 'program_genes.txt').read_text().splitlines())
    barcodes = pd.Index((SC_DIR / 'barcodes.txt').read_text().splitlines())
    meta_sc = pd.read_csv(SC_DIR / 'metadata.csv.gz', index_col=0, low_memory=False).loc[barcodes]
    naive = meta_sc['treatment'].eq('aPD-1 (naive)')
    patients = sorted(meta_sc.loc[naive & meta_sc['timepoint'].eq('Baseline'), 'patient_alias'].unique())
    selected = naive & meta_sc['patient_alias'].isin(patients) & meta_sc['timepoint'].isin(TP_ORDER)
    X = mmread(SC_DIR / 'program_expression.mtx').tocsr()[:, selected.to_numpy()]
    meta_sc = meta_sc.loc[selected]
    barcodes = meta_sc.index
    baseline_indices = {
        patient: np.flatnonzero(meta_sc['patient_alias'].eq(patient) & meta_sc['timepoint'].eq('Baseline'))
        for patient in patients
    }
    weights = prepare_weights(WEIGHTS_PATH)
    hcop = pd.read_csv(HCOP_PATH, sep='\t', compression='gzip', dtype=str, low_memory=False)
    hcop = hcop.loc[hcop['mouse_symbol'].isin(weights['mouse_symbol']), ['mouse_symbol', 'human_symbol']].dropna().drop_duplicates()
    mapping = weights.merge(hcop, how='left', on='mouse_symbol', validate='many_to_many')
    mapping['expression_present'] = mapping['human_symbol'].isin(genes)
    key = ['source_program', 'mouse_symbol']
    mapping['expressed_ortholog_count'] = mapping.groupby(key)['expression_present'].transform('sum')
    mapping['weight_share_before_coverage_normalization'] = np.where(mapping['expression_present'], mapping['mouse_weight_normalized'] / mapping['expressed_ortholog_count'].replace(0, np.nan), 0.0)
    coverage_mass = mapping.groupby('source_program')['weight_share_before_coverage_normalization'].transform('sum')
    mapping['weight_used_for_scoring'] = np.where(mapping['expression_present'], mapping['weight_share_before_coverage_normalization'] / coverage_mass, 0.0)
    print('Expression:', X.shape, 'Metadata:', meta_sc.shape)
    gene_idx = pd.Series(np.arange(len(genes)), index=genes)
    means, seconds = ([], [])
    for idx in baseline_indices.values():
        means.append(np.asarray(X[:, idx].mean(axis=1)).ravel())
        seconds.append(np.asarray(X[:, idx].power(2).mean(axis=1)).ravel())
    gene_mean = np.mean(means, axis=0)
    gene_sd = np.sqrt(np.maximum(np.mean(seconds, axis=0) - gene_mean ** 2, 0))
    source_scores = {}
    for source in SOURCE_TO_MODEL:
        used = mapping[(mapping['source_program'] == source) & mapping['expression_present']]
        w = used.groupby('human_symbol')['weight_used_for_scoring'].sum()
        rows = gene_idx.loc[w.index].to_numpy()
        keep = gene_sd[rows] > 1e-08
        w, rows = (w.iloc[np.flatnonzero(keep)], rows[keep])
        w = w / w.sum()
        coef = w.to_numpy() / gene_sd[rows]
        source_scores[source] = np.asarray(X[rows, :].T @ coef).ravel() - np.sum(w.to_numpy() * gene_mean[rows] / gene_sd[rows])
    src = pd.DataFrame(source_scores, index=barcodes)
    z_sc = pd.DataFrame(index=barcodes)
    z_sc['Precursor'] = src[['Naive-Like', 'Memory-Like', 'Naive-Il7r']].mean(axis=1)
    z_sc['Activation'] = src['Activation']
    z_sc['Early effector'] = src['Eff-Early']
    z_sc['Cytotoxic'] = src['Eff-Cytox']
    z_sc['Dysfunction'] = src['Dysf']
    means, seconds = ([], [])
    for idx in baseline_indices.values():
        x = z_sc.iloc[idx]
        means.append(x.mean().to_numpy())
        seconds.append((x ** 2).mean().to_numpy())
    mu = np.mean(means, axis=0)
    sd = np.sqrt(np.maximum(np.mean(seconds, axis=0) - mu ** 2, 0))
    z_sc[PROGRAM_ORDER] = (z_sc[PROGRAM_ORDER] - mu) / sd
    print('Treatment-naive baseline patients:', patients)
    rows = []
    for p in patients:
        s_bl = float(z_sc.iloc[baseline_indices[p]].mean() @ alpha)
        rows.append({'patient_alias': p, 'timepoint': 'Baseline', 'delta_CD8score': 0.0})
        for tp in TP_ORDER[1:]:
            mask = meta_sc['patient_alias'].eq(p) & meta_sc['timepoint'].eq(tp)
            if mask.sum() == 0:
                continue
            s_fu = float(z_sc.loc[mask, PROGRAM_ORDER].mean() @ alpha)
            rows.append({'patient_alias': p, 'timepoint': tp, 'delta_CD8score': s_fu - s_bl})
    human_cd8score = pd.DataFrame(rows)
    human_summary = human_cd8score.groupby('timepoint')['delta_CD8score'].agg(mean='mean', sd='std', n='count').reindex(TP_ORDER).reset_index()
    human_summary['sem'] = human_summary['sd'] / np.sqrt(human_summary['n'])
    mouse = pd.read_csv(MOUSE_PATH).sort_values(['treatment', 'sim_seed', 'time']).copy()
    mouse_cols = {'Precursor': 'P_precursor_zstd', 'Activation': 'Activation_zstd', 'Early effector': 'Eff_Early_zstd', 'Cytotoxic': 'Eff_Cytox_zstd', 'Dysfunction': 'Dysf_zstd'}
    mouse['CD8score'] = sum((mouse[mouse_cols[p]] * alpha[p] for p in PROGRAM_ORDER))
    mouse['delta_CD8score'] = mouse['CD8score'] - mouse.groupby(['treatment', 'sim_seed'])['CD8score'].transform('first')
    pd1 = mouse.loc[mouse['treatment'].eq('Anti-PD1'), ['sim_seed', 'time', 'delta_CD8score']].rename(columns={'delta_CD8score': 'pd1'})
    ctrl = mouse.loc[mouse['treatment'].eq('Control'), ['sim_seed', 'time', 'delta_CD8score']].rename(columns={'delta_CD8score': 'control'})
    mouse_specific = pd1.merge(ctrl, on=['sim_seed', 'time'])
    mouse_specific['PD1_specific_CD8score'] = mouse_specific['pd1'] - mouse_specific['control']
    mouse_specific_summary = mouse_specific.groupby('time')['PD1_specific_CD8score'].agg(mean='mean', sd='std', n='count').reset_index()
    mouse_specific_summary['sem'] = mouse_specific_summary['sd'] / np.sqrt(mouse_specific_summary['n'])
    human_cd8score.to_csv(OUT_DIR / 'human_patient_delta_cd8score.csv', index=False)
    human_summary.to_csv(OUT_DIR / 'human_delta_cd8score_summary.csv', index=False)
    mouse_specific_summary.to_csv(OUT_DIR / 'mouse_pd1_specific_delta_cd8score_summary.csv', index=False)
    print('Saved to:', OUT_DIR)
