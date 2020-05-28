This repository contains the code for the paper
"Active Causal Structure Learning via Directed Clique Trees"

The following command will set up a Python virtual environment
with the necessary dependencies installed.
```bash
bash setup.sh
```

To run the code, activate the virtual environment:
```bash
source venv/bin/activate
```

Then,
```bash
python3 fig1a.py
python3 fig1b.py
python3 fig1c.py
```

This will create two new directories. `data/` will contain synthetic DAGs and the results
(number of interventions used and computation time) for each intervention policy. `figures/`
will contain plots of these results as they appear in the paper.
