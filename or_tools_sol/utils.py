import pandas as pd 
from evaluation import change_selling_prices_format, get_actual_demand
import pdb
import ast
def read_input_data(datacenters, demand, selling_prices, servers):
    """
    Read the input data and return the datacenters, demand, selling prices and servers
    """
    datacenters = pd.read_csv(datacenters)
    demand = get_actual_demand(pd.read_csv(demand))
    ## select first 50 rows
    demand = demand.iloc[:350]
    selling_prices = change_selling_prices_format(pd.read_csv(selling_prices))
    servers = pd.read_csv(servers)
    ## process release_time
    servers['release_time'] = servers['release_time'].apply(lambda x: ast.literal_eval(x))

    return datacenters, demand, selling_prices, servers
