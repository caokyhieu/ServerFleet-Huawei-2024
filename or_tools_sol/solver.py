
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
        self.binary_lifespan_variables = {}

    
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
        elif time_step >= release_time[0] and time_step <= release_time[1]:
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
            self.variables[ts] = {}
            for server in servers:
                ## check the release time of the server
                check_release_time = self.check_release_time(server,ts)
                if check_release_time == 0:
                    ## do nothing
                    continue
                elif check_release_time == 1:
                    self.variables[ts][server] = {}
                    for dc in datacenters:
                        self.variables[ts][server][dc] = {}
                        if ts==1:
                            action = 'buy'
                            self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
                        else:
                            for action in actions:
                                self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
                else:
                    ## can not buy that server, but for other action is ok
                    self.variables[ts][server] = {}
                    for dc in datacenters:
                        self.variables[ts][server][dc] = {}
                        for action in actions:
                            if action == 'buy':
                                continue
                            else:
                                self.variables[ts][server][dc][action] = self.model.NewIntVar(0, self.max_server_buying, f"x_{ts}_{server}_{dc}_{action}")
            ## flatten the dictionary, seperate by "_"

            self.variables = pd.json_normalize(self.variables, sep='_').to_dict(orient='records')[0]

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
        self.dict_life_span = {}
        for server in servers:
            self.dict_life_span[server] = {}
            for dc in datacenters:
                buying_steps = self.get_timestep_buying_server(server, dc)
                buying_variables = self.get_buying_servers(server, dc)
                dismiss_variables = self.get_dismiss_servers(server, dc)
                ## start buying:
                start_time = buying_steps[0] - 1
                ## buying
                buying_variables = buying_variables[start_time:]
                ## dismiss
                dismiss_variables = dismiss_variables[start_time:]

                actual_time_step = len(time_steps) - start_time 
                assert(actual_time_step==len(buying_variables)), f'wrong steps, {len(buying_variables)}!= {actual_time_step}'

                # Initialize variables for life span tracking
                life_span = [self.model.NewIntVar(0, 10000 * MILLIONS, f'life_span_{t}_{server}_{dc}') for t in range(actual_time_step)]

                
                # This matrix will store the remaining servers from each buy at each timestep
                remaining_servers = [[self.model.NewIntVar(0, 100 * MILLIONS, f'remaining_servers_{i}_{t}_{server}_{dc}') for t in range(actual_time_step)] for i in range(actual_time_step)]
                # Initialize constraints
                for t in range(actual_time_step):
                    if t == 0:
                        # At t = 0, we only have the initial buy, no dismissals yet
                        self.model.Add(remaining_servers[0][t] == buying_variables[0])
                        for i in range(1, actual_time_step):
                            self.model.Add(remaining_servers[i][t] == 0)
                    else:
                        # Update remaining servers by considering buys and dismissals
                        for i in range(actual_time_step):
                            if i == t:
                                # New buy at this timestep, initialize remaining_servers
                                self.model.Add(remaining_servers[i][t] == buying_variables[t])
                            else:
                                # Calculate remaining servers by subtracting the dismissals
                                if t == i + 1:
                                    # First timestep after buy, subtract any dismissals
                                    self.model.Add(remaining_servers[i][t] == remaining_servers[i][t-1] - dismiss_variables[t-1])
                                elif t > i + 1:
                                    # Subsequent timesteps, just track remaining
                                    self.model.Add(remaining_servers[i][t] == remaining_servers[i][t-1])
                                    
                                # Ensure that servers dismissed are not negative
                                self.model.Add(remaining_servers[i][t] >= 0)

                    # Calculate the total life span at each timestep
                    self.model.Add(life_span[t] == sum((remaining_servers[i][t] * max(t - i + 1,0)) for i in range(actual_time_step)))
                ## augment the life_span arr to the length of time step
                aug_life_span = [0] * start_time  + life_span

                self.dict_life_span[server][dc] = aug_life_span
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
        

    def create_time_interval(self):
        """Create a time interval for each time step."""
        time_steps = self.demand.time_step.unique()
        # pdb.set_trace()
        ## create interval for each time step
        self.intervals = []
        for ts in time_steps:
            interval = self.model.NewIntervalVar(1, ts-1, ts, f"interval_{ts}")
            self.intervals.append(interval)
        pass
        

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
                            buy_var_is_zero = self.model.NewBoolVar(f"buy_var_is_zero_{_t}_{s}_{dc}")
                            dismiss_var_is_zero = self.model.NewBoolVar(f"dismiss_var_is_zero_{_t}_{s}_{dc}")

                            #Link the boolean variables with the integer variables
                            self.model.Add(buy_var == 0).OnlyEnforceIf(buy_var_is_zero)
                            self.model.Add(buy_var != 0).OnlyEnforceIf(buy_var_is_zero.Not())

                            self.model.Add(dismiss_var == 0).OnlyEnforceIf(dismiss_var_is_zero)
                            self.model.Add(dismiss_var != 0).OnlyEnforceIf(dismiss_var_is_zero.Not())

                            # At least one of the boolean variables must be True
                            self.model.AddBoolOr([buy_var_is_zero, dismiss_var_is_zero])

                            ## add constraint for dismiss
                            self.model.Add(dismiss_var <= sum_server_slot[dc][s])
                            ## add to sum_server
                            sum_server_slot[dc][s] += buy_var - dismiss_var
                        elif buy_action:
                            buy_var = self.get_variables(_t,s,dc,'buy')
                            sum_server_slot[dc][s] += buy_var 
                        elif dismiss_action:
                            ## get dissmed var
                            dismiss_var = self.get_variables(_t,s,dc,'dismiss')
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

    def compute_utilization_revenue(self,step=2,latency='low', server_generation='CPU.S1'):
        """
        - Integrate the utilization and revenue + datacenter constraints to reduce complexity
        """
        
        demand = self.demand[self.demand.time_step==step][latency].sum()
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
        scaled_Z = self.model.NewIntVar(1, 100* MILLIONS, f'scaled_Z_{step}_{latency}_{server_generation}')
        self.model.Add(scaled_Z == Z * scaled_float)
        self.model.AddDivisionEquality(scaled_Zf, scaled_Z, scaling_factor)
       
        # scaled_Zf += 1 ##
        min_zf_demand = self.model.NewIntVar(0, MILLIONS, f'min_zf_demand_u_{step}_{latency}_{server_generation}')
        self.model.AddMinEquality(min_zf_demand, [scaled_Zf, int(demand)])
        num = self.model.NewIntVar(0, MILLIONS, f'num_{step}_{latency}_{server_generation}')
        self.model.Add(num==min_zf_demand)
        denom = self.model.NewIntVar(1, MILLIONS, f'denom_{step}_{latency}_{server_generation}')
        self.model.Add(denom==scaled_Zf)
        utilization = self.model.NewIntVar(0, MILLIONS, f'utilization_{step}_{latency}_{server_generation}')
        self.model.AddDivisionEquality(utilization,num,denom)

        ## compute revenue
        p_ig = int(self.sellingprices.loc[server_generation][latency])
        revenue = min_zf_demand * p_ig  
                        
        return (utilization,revenue)
    

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
                total_servers += self.get_accum_deployed_server_variables(step, server, dc)
            ## normalize lifespan by life_expectancy
            normalized_life_span = self.model.NewIntVar(0, 100000 * MILLIONS, f"normalized_life_span_{server}_{step}")
            lifespan_var = self.model.NewIntVar(0, 100000 * MILLIONS, f"lifespan_var{server}_{step}")
            self.model.Add(lifespan_var==lifespan)
            self.model.AddDivisionEquality(normalized_life_span, lifespan_var, life_expectancy) 
            total_lifespan += normalized_life_span
        
        ## now add new varaibles
        normalized_life_span_t = self.model.NewIntVar(0 , MILLIONS, f"normalized_life_span_{step}")
        num_total_lifepsan = self.model.NewIntVar(0 , 100000 * MILLIONS, f"num_total_lifepsan_{step}")
        denom_total_servers = self.model.NewIntVar(1 , 100000 * MILLIONS, f"denom_total_servers_{step}")
        self.model.Add(num_total_lifepsan == total_lifespan)
        self.model.Add(denom_total_servers == total_servers + 1)
        self.model.AddDivisionEquality(normalized_life_span_t, num_total_lifepsan, denom_total_servers)
        return normalized_life_span_t

    def add_demand_constraints(self):
        """
        Methods to add the demand constraints for each time steps
        """
        pass

    def add_objective(self):
        """
        Methods to add the objective function
        """
        Oj = 0
        for i in self.demand.time_step.unique():
            revenue = 0
            utilization = 0
            for latency in ['low','medium','high']:
                for server_generation in self.servers.server_generation:
                    # bj_i = self.model.NewIntVar(0, 9999999999999, f'oj_{i}_{latency}')
                    # first_number = self.model.NewIntVar(0, 9999999999999, f'first_number_{i}_{latency}')
                    # second_number = self.model.NewIntVar(0, 9999999999999, f'second_number_{i}_{latency}')
                    # self.model.Add(first_number == self.compute_revenue(i,latency,server_generation))
                    # self.model.Add(second_number == self.compute_utilization(i,latency,server_generation))
                    # self.model.AddMultiplicationEquality(bj_i, first_number, second_number)
                    # pdb.set_trace()
                    u,r = self.compute_utilization_revenue(i,latency,server_generation)
                    # # pdb.set_trace()
                    revenue += r
                    utilization += u
            Oj += revenue + utilization
                    # Oj += self.compute_revenue(i,latency,server_generation)
            #         revenue += self.compute_revenue(i,latency,server_generation)
            #         utilization += self.compute_utilization(i,latency,server_generation)
            # temp_obj = self.model.NewIntVar(0, 9999999999999, f'temp_obj_{i}')
        self.model.maximize(Oj)
        # print(f"Solving with {self.model.SolverVersion()}")
    
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
        # self.create_auxillary_variables()
        self.add_objective()
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True  # Disable search logging, enable when debugging
        # solver.parameters.cp_model_probing_level = 0  # Set probing level to 0
        solver.parameters.cp_model_presolve = False  # Enable presolve to improve the performance
        solver.parameters.num_search_workers = 8  # Use 8 threads
        # solver.parameters.enumerate_all_solutions = True

        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH  # Use portfolio search ['AUTOMATIC_SEARCH', 'FIXED_SEARCH', 'LP_SEARCH', 'PORTFOLIO_SEARCH']
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
            print(f'{solver.SufficientAssumptionsForInfeasibility()}') 
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


def main():
    datacenters_path = './data/datacenters.csv'
    demand_path = './data/demand.csv'
    selling_prices_path = './data/selling_prices.csv'
    servers_path = './data/servers.csv'
    datacenters, demand, selling_prices, servers = read_input_data(datacenters_path, demand_path, selling_prices_path, servers_path)
    solver = ServerFleetSolver(datacenters, demand, selling_prices, servers)
    solver.solve()
    pass

