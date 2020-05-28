from submit.result_getter import ResultGetter
from submit.dag_loader import DagSampler
from submit.plot_results_vary_nnodes2 import plot_results_vary_nodes

algs = {
    'random',
    'dct',
    # 'opt_single',
    'coloring',
    # 'greedy_minmax',
    # 'greedy_entropy'
}
# nnodes_list = [50, 100, 150, 200]
# plot_results_vary_nodes(
#     nnodes_list,
#     10,
#     DagSampler.HAIRBALL_PLUS,
#     dict(e_min=2, e_max=5, degree=4),
#     algorithms=algs
# )

nnodes_list = [100, 200, 300]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS,
    dict(degree=4, nontree_factor=.02),
    algorithms=algs
)

# r = ResultGetter(
#     algs,
#     nnodes_list,
#     DagSampler.HAIRBALL_PLUS,
#     other_params_list=[dict(e_min=2, e_max=5, degree=3)],
#     ngraphs=10
# )
# res_df = r.get_results(overwrite=True)

