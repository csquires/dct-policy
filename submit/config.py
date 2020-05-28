import os
import seaborn as sns

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, 'data')
FIGURE_FOLDER = os.path.join(BASE_FOLDER, 'figures')

policies = [
    'dct',
    'random',
    'coloring',
    'opt_single',
    'greedy_minmax',
    'greedy_entropy'
]
POLICY2COLOR = dict(zip(policies, sns.color_palette('bright')))
POLICY2LABEL = {
    'dct': 'DCT',
    'random': 'Random',
    'coloring': 'Coloring',
    'opt_single': 'OptSingle',
    'greedy_minmax': 'GreedyMEC',
    'greedy_entropy': 'GreedyEntropy',
}
