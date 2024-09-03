from or_tools_sol.solver import ServerFleetSolver 
from or_tools_sol.utils import read_input_data
from seeds import known_seeds
import numpy as np


datacenters_path = './data/datacenters.csv'
demand_path = './data/demand.csv'
selling_prices_path = './data/selling_prices.csv'
servers_path = './data/servers.csv'

seeds = known_seeds('training')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)
    datacenters, demand, selling_prices, servers = read_input_data(datacenters_path, demand_path, selling_prices_path, servers_path)
    solver = ServerFleetSolver(datacenters, demand, selling_prices, servers)
    solver.solve(save_path=f"./output/{seed}.json")
