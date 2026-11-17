"""Export the frozen Top50 list used for S5b functional membership annotations."""
from common.annotation import validate_weights

def run(inputs, output):
    _, top50 = validate_weights(inputs.results / 'TCGA_SKCM/gene_level_cd8score/cd8score_gene_weights.csv')
    table = top50[['abs_model_weight_rank', 'human_ensembl_gene', 'human_symbol', 'cd8score_gene_weight', 'weight_direction']].copy()
    table.insert(0, 'selection_rule', 'Top50 by descending abs(cd8score_gene_weight)')
    table['selected_without_survival_outcome'] = True
    table.to_csv(output / 'top50_gene_sets.csv', index=False)
