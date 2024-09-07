
from ortools.linear_solver import pywraplp
from scipy.stats import truncweibull_min
from ortools.sat.python import cp_model
import pandas as pd
from or_tools_sol.utils import read_input_data
import pdb
import uuid
import json
from utils import save_solution
from collections import deque
import numpy as np
MILLIONS = 100000
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        for v in self.__variables:
            if self.value(v)!=0:
                print(f"{v}={self.value(v)}", end=" ")
        print()

    @property
    def solution_count(self) -> int:
        return self.__solution_count
    
class ServerFleetSolver:

    def __init__(self, datacenters, demand, sellingprices,servers):

        self.datacenters = datacenters
        self.demand = demand
        self.sellingprices = sellingprices
        self.servers = servers
        # self.solver = pywraplp.Solver.CreateSolver("SAT")
        self.model = cp_model.CpModel()
        self.actions = ['buy',
                        # 'move.DC1',
                        # 'move.DC2',
                        # 'move.DC3',
                        # 'move.DC4',
                        'dismiss'
                        ]
        self.max_server_buying = 200
        ## create nested dictionary for vriables -4 level

        self.variables = {}
        ## new  variables
        self.binary_action_variables = {}

    
    def check_release_time(self,server_generation, time_step):
        """
        Methods to check the release time of the servers
        Returns: 0 if less
                1 if in 
                2 if greater
        """
        ## get release time interval of the server
        release_time = self.servers[self.servers.server_generation==server_generation].release_time.values[0]
        if time_step < release_time[0]:
            return 0
        elif time_step == release_time[0]:
            return 3
        elif time_step > release_time[0] and time_step <= release_time[1]:
            return 1
        else:
            return 2

    def create_variables(self):
        """
        Methods to create all variables for each time steps, according demand, 
        datacenters constraints, the server  and the actions can be made.
        Dimension of the variables: time_steps * servers * datacenters * actions

        Especially, with the move will devide into 4 different datacenters
        Already have constriant for time_release

        """
        ## Assume we only have two actions [buy and dismiss]

        time_steps = self.demand.time_step.unique()
        servers = self.servers.server_generation
        datacenters = self.datacenters.datacenter_id

        actions = self.actions
        for ts in time_steps:
            self.binary_action_variables[ts] = {}
            self.variables[ts] = {}
            for server in servers:
                ## check the release time of the server
                check_release_time = self.check_release_time(server,ts)
                if check_release_time == 0:
                    ## do nothing
                    continue
                elif check_release_time == 3:
                    ## just can do buy
                    self.binary_action_variables[ts][server] = {}
                    self.variables[ts][server] = {}
                    for dc in datacenters:
                        self.binary_action_variables[ts][server][dc] = {}
                        self.variables[ts][server][dc] = {}
                        action = 'buy'
                        self.binary_action_variables[ts][server][dc][action] = self.model.NewBoolVar(f"{ts}_{server}_{dc}_{action}_is_not_zero")
                        self.model.Add(self.binary_action_variables[ts][server][dc][action]==True)
                        self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
                elif check_release_time == 1:
                    self.binary_action_variables[ts][server] = {}
                    self.variables[ts][server] = {}
                    for dc in datacenters:
                        self.binary_action_variables[ts][server][dc] = {}
                        self.variables[ts][server][dc] = {}
                        for action in actions:
                            self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
                            self.binary_action_variables[ts][server][dc][action] = self.model.NewBoolVar(f"{ts}_{server}_{dc}_{action}_is_not_zero")
                else:
                    ## can not buy that server, but for other action is ok
                    self.variables[ts][server] = {}
                    self.binary_action_variables[ts][server] = {}
                    for dc in datacenters:
                        self.variables[ts][server][dc] = {}
                        self.binary_action_variables[ts][server][dc] = {}
                        for action in actions:
                            if action == 'buy':
                                continue
                            else:
                                self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
                                self.binary_action_variables[ts][server][dc][action] = self.model.NewBoolVar(f"{ts}_{server}_{dc}_{action}_is_not_zero")
            ## flatten the dictionary, seperate by "_"

            self.variables = pd.json_normalize(self.variables, sep='_').to_dict(orient='records')[0]
            self.binary_action_variables = pd.json_normalize(self.binary_action_variables, sep='_').to_dict(orient='records')[0]

        pass

    def get_timestep_buying_server(self,server,dc):
        """
        Return tiemstep at this there is buy server (maybe equal 0)
        """
        time_steps = self.demand.time_step.unique()
        ts_arr = []
        for ts in time_steps:
            buy_exist = self.check_variables_exist(ts,server,dc,'buy')
            if buy_exist:
                ts_arr.append(ts)
        return ts_arr
    
    def get_buying_servers(self,server,dc):
        """
        Return buy servers at differnt time step
        """
        time_steps = self.demand.time_step.unique()
        results = []
        for ts in time_steps:
            buy_exist = self.check_variables_exist(ts,server,dc,'buy')
            if buy_exist:
                results.append(self.get_variables(ts,server,dc,'buy'))
            else:
                results.append(0)
        return results
    
    def get_dismiss_servers(self, server, dc):
        """
        Return dismiss servers at different time step
        """
        time_steps = self.demand.time_step.unique()
        results = []
        for ts in time_steps:
            dismiss_exist = self.check_variables_exist(ts,server,dc,'dismiss')
            if dismiss_exist:
                results.append(self.get_variables(ts,server,dc,'dismiss'))
            else:
                results.append(0)
        return results


    def create_auxillary_variables(self):
        """
        Methods to create the accilary variables
        Create a binary variable to indicate that the server is still alive or not

        """
        time_steps = self.demand.time_step.unique()
        servers = self.servers.server_generation
        datacenters = self.datacenters.datacenter_id
        self.dict_alive_servers = {}
        self.dict_life_span = {}
        for server in servers:
            self.dict_alive_servers[server] = {}
            self.dict_life_span[server] = {}
            for dc in datacenters:
                
                buying_variables = self.get_buying_servers(server, dc)
                dismiss_variables = self.get_dismiss_servers(server, dc)
                # Initialize variables for life span tracking
                life_span = [0 for _ in range(len(time_steps))]

                # This matrix will store the remaining servers from each buy at each timestep
                remaining_servers = [[0 for t in range(len(time_steps))] for i in range(len(time_steps))]
                true_remaining_servers = [[self.model.NewIntVar(0, 100 * self.max_server_buying, f"remain_servers_{server}_{dc}_{i}_{j}") for j in range(len(time_steps)) ]for i in range(len(time_steps))]
                
                negative_indicator = []
                for i in range(len(time_steps)):
                    row_indicator = [self.model.NewBoolVar(f"negative_indicator_{i}_{j}") for j in range(len(time_steps))]
                    ## add consecutive constraints
                    for j in range(1,len(time_steps) ):
                        self.model.Add(row_indicator[j - 1] <= row_indicator[j])
                    negative_indicator.append(row_indicator)
                
                    
                # Initialize constraints
                for i in range(len(remaining_servers)):
                    for j in range(len(remaining_servers)):
                        if i == j:
                            remaining_servers[i][j] = buying_variables[i]
                            self.model.Add( negative_indicator[i][j] == 0)
                        elif i>j:
                            remaining_servers[i][j] = 0
                            self.model.Add( negative_indicator[i][j] == 0)
                        else:
                            remaining_servers[i][j] = remaining_servers[i][j-1] - dismiss_variables[j]
                            self.model.Add( remaining_servers[i][j] < 0).OnlyEnforceIf(negative_indicator[i][j])
                            self.model.Add( remaining_servers[i][j] >=0 ).OnlyEnforceIf(negative_indicator[i][j].Not())
                        temp_var = self.model.NewIntVar(0, 100* self.max_server_buying,f"temp_var_{server}_{dc}_{i}_{j}")
                        self.model.Add(temp_var==remaining_servers[i][j])
                        self.model.AddMultiplicationEquality(true_remaining_servers[i][j],negative_indicator[i][j], temp_var )
                    for row in range(i+1):
                        life_span[i] += true_remaining_servers[row][i] * (1 + i - row)               
                self.dict_life_span[server][dc] = life_span
                self.dict_alive_servers[server][dc] = true_remaining_servers
        pass

        
    def get_accum_deployed_server_variables(self,ts,server,dc):
        "Return accumulated server variables"
        results = 0
        
        for i in range(1,ts+1):

            buy_exist = self.check_variables_exist(i , server, dc, 'buy')
            dismiss_exist = self.check_variables_exist(i, server, dc, 'dismiss')

            if buy_exist and dismiss_exist:
                results += self.get_variables(i , server, dc,'buy') - self.get_variables(i , server, dc,'dismiss')
            elif buy_exist:
                results += self.get_variables(i , server, dc, 'buy')
            elif dismiss_exist:
                results -= self.get_variables(i , server, dc, 'dismiss')
            else:
                continue
        return results
        

        

    def check_variables_exist(self,ts,server,dc,action):
        """
        Methods to check if the variables is exist in the variables dictionary
        """
        keys = f"{ts}_{server}_{dc}_{action}"
        if keys in self.variables.keys():
            return True
        else:
            return False
        
    def get_variables(self,ts,server,dc,action):
        """
        Methods to get the variables
        """
        keys = f"{ts}_{server}_{dc}_{action}"
        return self.variables[keys]
    
    def get_binary_action_variable(self,ts,server,dc, action):
        """
        Methods to get  binary variable
        """
        keys = f"{ts}_{server}_{dc}_{action}"
        if keys in self.binary_action_variables.keys():
            return self.binary_action_variables[keys]
        else:
            raise ValueError(f"Variable {keys} does not exist")

    def add_datacenters_capacity_constraint(self):
        """
        - sum of servers in each datacenter should be less than the capacity of the datacenter
        """
        
        time_steps = self.demand.time_step.unique()
        servers = self.servers.server_generation
        datacenters = self.datacenters.datacenter_id

        for ts in time_steps:
            sum_server_slot = {}
            for dc in datacenters:
                sum_server_slot[dc] = {}
                for s in servers:
                    sum_server_slot[dc][s] = 0
                        
            for dc in datacenters:
            ## capacity of this dc
                dc_capacity = self.datacenters[self.datacenters.datacenter_id == dc].slots_capacity.values[0]
                # pdb.set_trace()
                
                for _t in range(1,ts+1):
                    # for action in actions:
                    for s in servers:
                        ## assume that we just have two actions
                        buy_action = self.check_variables_exist(_t,s,dc,'buy')
                        dismiss_action = self.check_variables_exist(_t,s,dc,'dismiss')
                        if buy_action and dismiss_action:
                            ## get both variables
                            buy_var = self.get_variables(_t,s,dc,'buy')
                            dismiss_var = self.get_variables(_t,s,dc,'dismiss')
                            buy_var_is_not_zero = self.get_binary_action_variable(_t,s,dc,'buy')
                            dismiss_var_is_not_zero = self.get_binary_action_variable(_t,s,dc,'dismiss')

                            #Link the boolean variables with the integer variables
                            self.model.Add(buy_var == 0).OnlyEnforceIf(buy_var_is_not_zero.Not())
                            self.model.Add(buy_var != 0).OnlyEnforceIf(buy_var_is_not_zero)

                            self.model.Add(dismiss_var == 0).OnlyEnforceIf(dismiss_var_is_not_zero.Not())
                            self.model.Add(dismiss_var != 0).OnlyEnforceIf(dismiss_var_is_not_zero)

                            # At least one of the boolean variables must be True
                            self.model.AddBoolOr([buy_var_is_not_zero.Not(),dismiss_var_is_not_zero.Not()])

                            ## add constraint for dismiss
                            self.model.Add(dismiss_var <= sum_server_slot[dc][s])
                            ## add to sum_server
                            sum_server_slot[dc][s] += buy_var - dismiss_var
                        elif buy_action:
                            buy_var = self.get_variables(_t,s,dc,'buy')
                            buy_var_is_not_zero = self.get_binary_action_variable(_t,s,dc,'buy')
                            ## link the boolean variables with the integer variables
                            self.model.Add(buy_var == 0).OnlyEnforceIf(buy_var_is_not_zero.Not())
                            self.model.Add(buy_var != 0).OnlyEnforceIf(buy_var_is_not_zero)
                            ## add to sum_server
                            sum_server_slot[dc][s] += buy_var 
                        elif dismiss_action:
                            ## get dissmed var
                            dismiss_var = self.get_variables(_t,s,dc,'dismiss')
                            dismiss_var_is_not_zero = self.get_binary_action_variable(_t,s,dc,'dismiss')
                            ## link the boolean variables with the integer variables
                            self.model.Add(dismiss_var == 0).OnlyEnforceIf(dismiss_var_is_not_zero.Not())
                            self.model.Add(dismiss_var != 0).OnlyEnforceIf(dismiss_var_is_not_zero)
                            ## add cosntraint that sever dismissed must smaller than the number of this servers on this datacenter
                            self.model.Add(dismiss_var <= sum_server_slot[dc][s])
                            sum_server_slot[dc][s] -= dismiss_var

                        else:
                            continue

                ## calcluate the sum of server in this dc
                total_slot = 0
                for s in servers:
                    total_slot += sum_server_slot[dc][s] * self.servers[self.servers.server_generation==s].slots_size.values[0]
                self.model.Add(total_slot <= dc_capacity)
                        
        pass

    def add_life_span_constraint(self):
        """
        Methods to add the life span constraint
        This version only count dismiss and buy
        """

        time_steps = self.demand.time_step.unique()
        servers = self.servers.server_generation
        datacenters = self.datacenters.datacenter_id
        # actions = self.actions
        for ts in time_steps:
            ts_buyed_server = {}
            ## loop all servers
            for dc in datacenters:
                ts_buyed_server[dc] = {}
                for s in servers:
                    ## check if the buy variables is exist
                    buy_exist = self.check_variables_exist(ts,s,dc,'buy')
                    if buy_exist:
                        ts_buyed_server[dc][s] = self.get_variables(ts,s,dc,'buy')
                    else:
                        continue
                    ## get life span of the server
                    life_span = int(self.servers[self.servers.server_generation == s].life_expectancy.values[0])
                    dismiss_server = 0
                    if ts + life_span <= time_steps[-1]:
                        ## loop all time steps in this range
                        for _t in range(ts+1, ts+1+life_span):
                            ## check if the dismiss variables is exist
                            dismiss_exist = self.check_variables_exist(_t,s,dc,'dismiss')
                            if dismiss_exist:
                                dismiss_server += self.get_variables(_t,s,dc,'dismiss')
                            else:
                                continue
                        ## add constraint that the selled server has to larger than buyed server
                        self.model.Add(dismiss_server >= ts_buyed_server[dc][s])
        pass

    def compute_utilization_revenue(self,step=2,latency='low', server_generation='CPU.S1'):
        """
        - Integrate the utilization and revenue + datacenter constraints to reduce complexity
        """
        
        demand = self.demand[self.demand.time_step==step][self.demand.server_generation==server_generation][latency].sum()
        if demand==0:
            return 0,0
        # pdb.set_trace()
        ## now get the deployed servers
        t = 0
        deployed_servers = 0
        for i in range(1, step + 1):
            if latency== 'low':
                datacenters = ['DC1']
            elif latency== 'medium':
                datacenters = ['DC2']
            elif latency== 'high':
                datacenters = ['DC3','DC4']
            for dc in datacenters:
                ## check if the variables is exist
                buy_exist = self.check_variables_exist(i,server_generation,dc,'buy')
                dismiss_exist = self.check_variables_exist(i,server_generation,dc,'dismiss')
                if  buy_exist and  dismiss_exist:
                    t+=1
                    deployed_servers += self.get_variables(i,server_generation,dc,'buy') - self.get_variables(i,server_generation,dc,'dismiss')
                elif buy_exist:
                    t+=1
                    deployed_servers += self.get_variables(i,server_generation,dc,'buy')
                elif dismiss_exist:
                    t+=1
                    deployed_servers -= self.get_variables(i,server_generation,dc,'dismiss')
                else:
                    continue
        if t==0:
            return 0,0
                # deployed_servers += self.variables[i][server_generation][dc].get('buy',0)  - self.variables[i][server_generation][dc].get('dismiss',0)
        Z = deployed_servers * int(self.servers[self.servers.server_generation == server_generation].capacity.values[0])
        success_rate = 1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()
        scaled_float = int(success_rate * 100)
        scaling_factor = 100

        # Define integer variables for use in constraints
        # Zf = self.model.NewIntVar(1, MILLIONS, f'Zf_{step}_{latency}_{server_generation}')
        scaled_Zf = self.model.NewIntVar(1, MILLIONS, f'scaled_Zf_{step}_{latency}_{server_generation}')
        
        # Properly handle the scaled multiplication using integer logic
        scaled_Z = self.model.NewIntVar(1, 100 * MILLIONS, f'scaled_Z_{step}_{latency}_{server_generation}')
        self.model.Add(scaled_Z == Z * scaled_float)
        self.model.AddDivisionEquality(scaled_Zf, scaled_Z, scaling_factor)
       
        # scaled_Zf += 1 ##
        min_zf_demand = self.model.NewIntVar(0, MILLIONS, f'min_zf_demand_u_{step}_{latency}_{server_generation}')
        self.model.AddMinEquality(min_zf_demand, [scaled_Zf, int(demand)])
        num = self.model.NewIntVar(0, MILLIONS, f'num_{step}_{latency}_{server_generation}')
        self.model.Add(num==min_zf_demand * scaling_factor)
        denom = self.model.NewIntVar(1, MILLIONS, f'denom_{step}_{latency}_{server_generation}')
        self.model.Add(denom==scaled_Zf)
        utilization = self.model.NewIntVar(0, scaling_factor, f'utilization_{step}_{latency}_{server_generation}')
        self.model.AddDivisionEquality(utilization,num,denom)

        ## compute revenue
        p_ig = int(self.sellingprices.loc[server_generation][latency])
        revenue = min_zf_demand * p_ig  
                        
        return (utilization,revenue)

    

    def compute_normalized_lifespan(self,step=2):
        """
        Methods to compute the normalize lifespan of the servers
        L = 1/|S| * \sum_{s} x_{s}/\hat{x_{s}}
        """
        servers = self.servers.server_generation
        datacenters = self.datacenters.datacenter_id
        total_lifespan = 0
        total_servers = 0
        ## loop over servers
        for server in servers:
            ## get life expectancy
            life_expectancy = int(self.servers[self.servers.server_generation == server].life_expectancy.values[0])
            lifespan = 0
            ## loop over datacenters
            for dc in datacenters:
                ## get life span for corresponding server from datacenters
                try:
                    lifespan += self.dict_life_span[server][dc][step - 1]
                except:
                    print(f"out of range: step {step} server {server} dc {dc} , but the array only have {len(self.dict_life_span[server][dc])}")
                ## get toal deploy for corresponding server from datacenters
                acc_servers = self.get_accum_deployed_server_variables(step, server, dc)
                total_servers += acc_servers
                ## add constraint
                self.model.Add(lifespan <= acc_servers * life_expectancy)
            ## normalize lifespan by life_expectancy
            normalized_life_span = self.model.NewIntVar(0, 100000 * MILLIONS, f"normalized_life_span_{server}_{step}")
            lifespan_var = self.model.NewIntVar(0, 100000 * MILLIONS, f"lifespan_var{server}_{step}")
            self.model.Add(lifespan_var==lifespan)
            self.model.AddDivisionEquality(normalized_life_span, lifespan_var, life_expectancy) 
            total_lifespan += normalized_life_span
        
        ## now add new variables
        normalized_life_span_t = self.model.NewIntVar(0 , MILLIONS, f"normalized_life_span_{step}")
        num_total_lifepsan = self.model.NewIntVar(0 , 100000 * MILLIONS, f"num_total_lifepsan_{step}")
        denom_total_servers = self.model.NewIntVar(1 , 100000 * MILLIONS, f"denom_total_servers_{step}")
        self.model.Add(num_total_lifepsan == total_lifespan)
        self.model.Add(denom_total_servers == total_servers)
        self.model.AddDivisionEquality(normalized_life_span_t, num_total_lifepsan, denom_total_servers)
        return normalized_life_span_t
    
    def compute_cost(self, steps=2, server='CPU.S1', dc='DC1'):
        """Compute the cost of the servers"""
        total_cost = 0
        ## take the life expectancy
        life_expectancy = int(self.servers[self.servers.server_generation == server].life_expectancy.values[0])
       
        ## check if there is any new buying machine right at this step
        buy_exist = self.check_variables_exist(steps,server,dc,'buy')
        
        if buy_exist:
            ## get binary variable
            buy_var = self.get_binary_action_variable(steps,server,dc,'buy')
            ## get the number of servers
            num_servers = self.get_variables(steps,server,dc,'buy')
            ## get the cost of the server
            cost = int(self.servers[self.servers.server_generation == server].purchase_price.values[0])
            ## create a new variable that the multiplied between bianry varaible and num_servers
            actual_num_servers = self.model.NewIntVar(0, 100 * self.max_server_buying, f"actual_num_servers_{steps}_{server}_{dc}")
            self.model.AddMultiplicationEquality(actual_num_servers, buy_var, num_servers)
            ## add the cost to total cost
            total_cost += actual_num_servers * cost
        ## calculate energy cost
        e_cost = int(self.servers[self.servers.server_generation == server].energy_consumption.values[0] * self.datacenters[self.datacenters.datacenter_id == dc].cost_of_energy.values[0]) 
        total_cost += e_cost
        ## take remain servers matrix
        remain_servers = self.dict_alive_servers[server][dc]
        ## loop over all time steps
        for i in range(steps):
            ## get the number of servers at this time step
            num_active_servers = remain_servers[i][steps - 1]
            ## get the maintainance cost of the server
            maintainance_cost = int(self.servers[self.servers.server_generation == server].average_maintenance_fee.values[0])
            ## indivual time span at this row
            individual_time_span = steps - i 
            ## individual alpha
            alpha_i = maintainance_cost * ( 1+ 1.5 *individual_time_span / life_expectancy *  np.log2(1.5 * individual_time_span / life_expectancy))
            ## multiply for 1000 and round it to nearest integer
            alpha_i = int(alpha_i * 100)
            # ## create new variable and devide it for 1000
            # alpha_i_var = self.model.NewIntVar(0, 1000 * MILLIONS, f"alpha_i_{steps}_{server}_{dc}_{i}")
            # self.model.AddDivisionEquality(alpha_i_var, alpha_i, 1000)
            ## create new variable that multiply between num_active_servers and alpha_i
            sub_var = self.model.NewIntVar(0, 1000 * MILLIONS, f"sub_var_{steps}_{server}_{dc}_{i}")
            self.model.AddMultiplicationEquality(sub_var, num_active_servers, alpha_i)
            total_cost +=  sub_var
            
        return total_cost


    def add_objective(self):
        """
        Methods to add the objective function
        """
        Oj = 0
        for i in self.demand.time_step.unique():
            revenue = 0
            utilization = 0
            cost=  0
            for latency in ['low','medium','high']:
                for server_generation in self.servers.server_generation:
                    u,r = self.compute_utilization_revenue(i,latency,server_generation)
                    # # pdb.set_trace()
                    revenue += r
                    utilization += u
            
            ## loop server and dc
            for server in self.servers.server_generation:
                for dc in self.datacenters.datacenter_id:
                    cost += self.compute_cost(i,server,dc)
            P = revenue - cost 
            U = utilization
            L = self.compute_normalized_lifespan(i)
           
            Oj += P + L + U
           
        self.model.maximize(Oj)
    
    def random_uud_server_id(self):
        """
        Methods to generate random server id
        """
        ## geneerate unique id
        return str(uuid.uuid4())
        


    def parse_solution(self,solver):
        """
        Methods to parse the solution
        like in solution_example.json
        [{"time_step": 1, "datacenter_id": "DC1", "server_generation": "CPU.S1", "server_id": "7ee8a1de-b4b8-4fce-9bd6-19fdf8c1e409", "action": "buy"}, ...]
        """
        results = []
        bought_servers = {}
        for dc in self.datacenters.datacenter_id:
            bought_servers[dc] = {}
            for server in self.servers.server_generation:
                bought_servers[dc][server] = []
        # pdb.set_trace()
        for ts in self.demand.time_step.unique():
            for dc in self.datacenters.datacenter_id:
                for server in self.servers.server_generation:
                    for action in self.actions:
                        if action == 'buy':
                            ## check if variables is exist
                            var_exists = self.check_variables_exist(ts,server,dc,action)
                            if not var_exists:
                                continue
                            var = self.get_variables(ts,server,dc,action)
                            if solver.Value(var)>0:
                                for _ in range(solver.Value(var)):
                                    new_server = {"time_step": int(ts), "datacenter_id": dc, "server_generation": server, "server_id": self.random_uud_server_id(), "action": action}
                                    results.append(new_server)
                                    bought_servers[dc][server].append(new_server)
                        elif action == 'dismiss':
                            var_exists = self.check_variables_exist(ts,server,dc,action)
                            if not var_exists:
                                continue
                            var = self.get_variables(ts,server,dc,action)
                            ## remove the newest bought

                            if solver.Value(var)>0:
                                for _ in range(solver.Value(var)):
                                    dismissed_server = bought_servers[dc][server].pop(0)
                                    results.append({"time_step": int(ts), "datacenter_id": dc, "server_generation": server, "server_id": dismissed_server["server_id"], "action": action})
                        elif action.startswith('move'):
                            var_exists = self.check_variables_exist(ts,server,dc,action)
                            if not var_exists:
                                continue
                            var = self.get_variables(ts,server,dc,action)

                            if solver.Value(var)>0:
                                results.append({"time_step": ts, "datacenter_id": dc, "server_generation": server, "server_id": self.random_uud_server_id(), "action": action})
                        else:
                            pass
        
        return results
        

    def solve(self,save_path='solution.json'):
        """
        Methods to solve the problem
        """
        self.create_variables()
        ## add intervals
        # self.create_time_interval()
        ## add datacenters constraints
        self.add_datacenters_capacity_constraint()
        # self.add_new_datacenters_capacity_constraint()
        ## add life span constraints
        self.add_life_span_constraint()
        ## add auxiliary varaibles
        self.create_auxillary_variables()
        self.add_objective()
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True  # Disable search logging, enable when debugging
        # solver.parameters.cp_model_probing_level = 0  # Set probing level to 0
        solver.parameters.cp_model_presolve = True  # Enable presolve to improve the performance
        solver.parameters.num_search_workers = 8  # Use 8 threads
        # solver.parameters.enumerate_all_solutions = True

        solver.parameters.search_branching = cp_model.LP_SEARCH  # Use portfolio search ['AUTOMATIC_SEARCH', 'FIXED_SEARCH', 'LP_SEARCH', 'PORTFOLIO_SEARCH']
        solver.parameters.max_time_in_seconds = 300.0  # Set a 120-second time limit
        solver.parameters.random_seed = 43  # Set a random seed for reproducibility
        
        # solution_printer = VarArraySolutionPrinter([v for n,v in self.variables.items()])
        validation_result = self.model.Validate()
        print(f"Validation result: {validation_result}")
        # pdb.set_trace()
        status = solver.solve(self.model)
        
        if status == cp_model.OPTIMAL:
            print("Solution:")
            print("Objective value =", solver.objective_value)
        elif status == cp_model.INFEASIBLE:
            for i in solver.SufficientAssumptionsForInfeasibility():
                print(self.model.VarIndexToVarProto(i).name)
            print("Model is infeasible.")
            print(solver.ResponseStats())
        
           
        else:
            print("The problem does not have an optimal solution.")

        print("\nStatistics")
        print(f"  status    : {solver.status_name(status)}")
        print(f"  conflicts : {solver.num_conflicts}")
        print(f"  branches  : {solver.num_branches}")
        print(f"  wall time : {solver.wall_time} s")
       
        results = self.parse_solution(solver)
        results = pd.DataFrame(results)
        save_solution(results,save_path)
        # results.to_json(save_path,orient='records')
        # with open(save_path, 'w') as f:
        #     json.dump(results, f)


