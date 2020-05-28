import whynot as wn

state = wn.hiv.State()
config = wn.hiv.Config()

run = wn.hiv.simulate(state, config)
graph = wn.causal_graphs.build_dynamics_graph(wn.hiv, [run], config)
