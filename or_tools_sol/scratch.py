import numpy as np
from scipy.stats import truncweibull_min
import uuid
import pdb

class Server:
    def __init__(self, id, server_generation, 
                 capacity, start_time, life_expectancy,
                 release_time, purchase_price,
                 slots_size, energy_consumption,
                 cost_of_moving, average_maintenance_fee):
        self.id = id
        self.capacity = capacity
        self.server_generation = server_generation
        self.start_time = start_time
        self.life_expectancy = life_expectancy
        self.release_time = release_time
        self.purchase_price = purchase_price
        self.slots_size = slots_size
        self.energy_consumption = energy_consumption
        self.cost_of_moving = cost_of_moving
        self.average_maintenance_fee = average_maintenance_fee

        assert self.start_time >= self.release_time[0], f" starting time {self.start_time} should be greater than the release time of the server  {self.release_time[0]}"
        assert self.start_time <= self.release_time[1], f" starting time {self.start_time} should be less than the release time of the server  {self.release_time[1]}"

    def __str__(self):
        return f"Server {self.id}, generation {self.server_generation} with capacity {self.capacity}"
    
    def __repr__(self):
        return self.__str__()
    
    def calculate_working_time(self, current_time):

        return current_time - self.start_time + 1
    
    def calculate_maintance_cost(self, current_time):
        working_time = self.calculate_working_time(current_time)
        maintance_cost=  self.average_maintenance_fee * ( 1+ 1.5 * working_time/self.life_expectancy * np.log2(1.5 * working_time/self.life_expectancy))
        if working_time ==1:
            ## add the purchased price
            maintance_cost += self.purchase_price
        return maintance_cost
    
    def check_server_valid(self, current_time):
        if self.start_time > current_time:
            return False
        if self.start_time + self.life_expectancy <= current_time:
            return False
        return True
    
    def calculate_normalized_timespan(self, current_time):
        return self.calculate_working_time(current_time) / self.life_expectancy
    

class DataCenter:
    def __init__(self, datacenter_id, servers, cost_of_energy, latency_sensitivity,slots_capacity):
        self.datacenter_id = datacenter_id
        self.servers = servers
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

    def __str__(self):
        return f"DataCenter {self.datacenter_id} with {len(self.servers)} servers"
    
    def __repr__(self):
        return self.__str__()
    
    def add_server(self, server):
        ##check that the datacenter not full capacity
        assert self.calculate_capacity(server.start_time) + server.capacity <= self.slots_capacity, f"Datacenter {self.datacenter_id} is full"
        self.servers.append(server)
    
    def remove_server(self, server):
        self.servers.remove(server)

    def pop_server(self, server_idx):
        return self.servers.pop(server_idx)
    
    def calculate_capacity(self, current_time):
        return sum([server.capacity for server in self.servers if server.check_server_valid(current_time)])
    
    def calculate_energy_consumption(self, current_time):
        return sum([server.energy_consumption * self.cost_of_energy for server in self.servers if server.check_server_valid(current_time)])
    
    def calculate_maintainance_cost(self, current_time):
        return sum([server.calculate_maintance_cost(current_time) for server in self.servers if server.check_server_valid(current_time)])
    
    def calculate_cost(self, current_time):
        cost = self.calculate_energy_consumption(current_time) + self.calculate_maintainance_cost(current_time)
        return cost
    
    def calculate_normalized_timespan(self, current_time):
        return sum([server.calculate_normalized_timespan(current_time) for server in self.servers if server.check_server_valid(current_time)])
    
    def calculate_capacity_server(self, current_time, server_generation):
        return sum([server.capacity for server in self.servers if server.check_server_valid(current_time) and (server.server_generation == server_generation)])
    
    def calculate_active_servers(self, current_time):
        return sum([1 for server in self.servers if server.check_server_valid(current_time)])
    
    def calculate_remained_slots(self, current_time):
        total_capacity = self.calculate_capacity(current_time)
        return self.slots_capacity - total_capacity
    


class Resource:

    def __init__(self, datacenters, selling_prices):
        self.datacenters = datacenters
        self.selling_prices = selling_prices

    def calculate_capacity(self, current_time, server_generation, latency_sensitivity):
        return sum([datacenter.calculate_capacity_server(current_time, server_generation) for datacenter in self.datacenters if datacenter.latency_sensitivity == latency_sensitivity])
    
    def calculate_utilization(self, current_time, server_generation, latency_sensitivity, demand):
        capacity = self.calculate_capacity( current_time, server_generation, latency_sensitivity)
        success_rate = 1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()
        capacity = success_rate * capacity
        if capacity==0:
            return 0
        return min(demand, capacity) / capacity
    
    def get_selling_price(self, server_generation, latency_sensitivity):
        # pdb.set_trace()
        return self.selling_prices[latency_sensitivity].loc[server_generation]
    
    def calculate_revenue(self, current_time, server_generation, latency_sensitivity, demand):
        capacity = self.calculate_capacity( current_time, server_generation, latency_sensitivity)
        success_rate = 1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()
        capacity = success_rate * capacity
        return min(demand, capacity) * self.get_selling_price(server_generation=server_generation, latency_sensitivity=latency_sensitivity)
    
    def calculate_cost(self, current_time):
        cost = 0
        for datacenter in self.datacenters:
            cost += datacenter.calculate_cost(current_time)
        return cost
    
    def calculate_normalized_timespan(self, current_time):
        denominator = self.calculate_active_servers(current_time)
        if denominator == 0:
            return 0
        return sum([datacenter.calculate_normalized_timespan(current_time) for datacenter in self.datacenters])/self.calculate_active_servers(current_time)
    
    
    def calculate_active_servers(self, current_time):
        return sum([dc.calculate_active_servers(current_time) for dc in self.datacenters])
    
    def get_datacenter(self, datacenter_id,return_idx=False):

        for i,dc in enumerate(self.datacenters):
            if dc.datacenter_id==datacenter_id:
                if not return_idx:
                    return self.datacenters.pop(i)
                else:
                    return i, self.datacenters.pop(i)
        raise ValueError(f"{datacenter_id} not in datacenters")
    
    def put_datacenters(self,datacenter,idx=None):
        if idx is not None:
            self.datacenters.insert(idx,datacenter)
        else:
            self.datacenters.append(datacenter)

    def calculate_remained_slots_for_datacenter(self, current_time, datacenter_id):
        for datacenter in self.datacenters:
            if datacenter.datacenter_id == datacenter_id:
                return datacenter.calculate_remained_slots(current_time)
        raise ValueError(f"{datacenter_id} not in datacenters")

       
    
    def calculate_remained_slots_for_latency(self, current_time, latency_sensitivity):
        return sum([datacenter.calculate_remained_slots(current_time) for datacenter in self.datacenters if datacenter.latency_sensitivity == latency_sensitivity])
    
    def calculate_remained_slots(self, current_time):
        return sum([datacenter.calculate_remained_slots(current_time) for datacenter in self.datacenters])
   
class Agent:

    def __init__(self,datacenters, demand, selling_prices, servers):
        self.datacenters = datacenters
        self.demand = demand
        self.selling_prices = selling_prices
        self.servers = servers
        # self.current_time = 0
        # self.resource = Resource(datacenters, selling_prices)
        ## create datacenters
        self._datacenters = []
        for _,row in self.datacenters.iterrows():
            dc = DataCenter(datacenter_id=row.datacenter_id, servers=[], cost_of_energy=int(row.cost_of_energy), 
                            latency_sensitivity=row.latency_sensitivity,slots_capacity=int(row.slots_capacity))
            self._datacenters.append(dc)

        self.resource = Resource(datacenters=self._datacenters, selling_prices=selling_prices)
        self.datacenter_ids = self.datacenters.datacenter_id.values
        self.latency_sensitivities = self.datacenters.latency_sensitivity.unique()
        self.server_generations = self.servers.server_generation.unique()

    def generate_unique_id(self):
        return str(uuid.uuid4())
    
    def create_server(self, server_generation, start_time):
        server = self.servers[self.servers.server_generation == server_generation]
        return Server(id=self.generate_unique_id(), server_generation=server.server_generation.values[0], 
                      capacity=server.capacity.values[0], start_time=start_time,
                      life_expectancy=server.life_expectancy.values[0], 
                      release_time=server.release_time.values[0], 
                      purchase_price=server.purchase_price.values[0],
                      slots_size=server.slots_size.values[0], 
                      energy_consumption=server.energy_consumption.values[0], 
                      cost_of_moving=server.cost_of_moving.values[0],
                      average_maintenance_fee=server.average_maintenance_fee.values[0])


    def add_server(self, start_time ,server_generation, datacenter_id):
        succeeded = False
        datacenter = self.resource.get_datacenter(datacenter_id)
        try:
            server = self.create_server(server_generation, start_time)
            datacenter.add_server(server)
            succeeded = True
        except Exception as e:
            # print("Error adding server", e)
            pass
            # print(f"Can not add server {server_generation} to datacenter {datacenter_id}")
        self.resource.put_datacenters(datacenter)
        return succeeded

    def remove_server(self, server, datacenter_id):
        i,datacenter = self.resource.get_datacenter(datacenter_id,return_idx=True)
        datacenter.remove_server(server)
        self.resource.put_datacenters(datacenter,idx=i)

    def calculate_cost(self, current_time):
        return self.resource.calculate_cost(current_time)
    
    def calculate_cost_for_datacenter(self, current_time, datacenter_id):
        for datacenter in self.resource.datacenters:
            if datacenter.datacenter_id == datacenter_id:
                return datacenter.calculate_cost(current_time)
        raise ValueError(f"{datacenter_id} not in datacenters")
    
    def get_datacenters_for_latency(self, latency_sensitivity):
        return self.datacenters[self.datacenters.latency_sensitivity == latency_sensitivity].datacenter_id.tolist()

    
    def calculate_revenue(self, current_time, demand):
        revenue = 0
        for server_generation in self.server_generations:
            for latency_sensitivity in self.latency_sensitivities:
                revenue += self.resource.calculate_revenue(current_time, server_generation, latency_sensitivity, 
                                                           demand[demand.server_generation==server_generation][latency_sensitivity].sum())
        return revenue
    
    def calculate_utilization(self, current_time, demand):
        utilization = 0
        for server_generation in self.server_generations:
            for latency_sensitivity in self.latency_sensitivities:
                utilization += self.resource.calculate_utilization(current_time, server_generation, latency_sensitivity, 
                                                                   demand[demand.server_generation==server_generation][latency_sensitivity].sum())
        return utilization/(len(self.server_generations) * len(self.latency_sensitivities))
    

    def calculate_normalized_timespan(self, current_time):
        return self.resource.calculate_normalized_timespan(current_time)
    
    def calculate_profit(self, current_time, demand):
        return self.calculate_revenue(current_time, demand) - self.calculate_cost(current_time)
    
    def calculate_objective(self, current_time, demand):
        return self.calculate_profit(current_time,demand) *  self.calculate_normalized_timespan(current_time) * self.calculate_utilization(current_time, demand)
        
    def get_demand(self, current_time):
        if isinstance(current_time,np.int64):
            return self.demand[self.demand.time_step == current_time]
        elif isinstance(current_time,list):
            d = self.demand[self.demand.time_step.isin(current_time)].groupby('server_generation').sum().reset_index()
            return d
    
    

    def strategy(self, demand, current_time):
        """With the demand, our strategy  will be greedy, add up untill match the demand
        
        Demand :    server_generation high low medium
                       CPU.S1         1     2    3   
        """
        ## first, loop row by row demand
        for _,row in demand.iterrows():
            ## get the server generation
            server_generation = row.server_generation
            server_capacity = self.servers[self.servers.server_generation==server_generation].capacity.values[0]
            ## get the latency sensitivity
            for latency in self.latency_sensitivities:
                ## get the demand
                demand = row[latency]
                # ## check current capcity for this latency of resource
                current_capacity = self.resource.calculate_capacity(current_time, server_generation, latency)
                if current_capacity >= demand:
                    continue
                ## substract the demand for current capacity
                demand -= current_capacity
                ## get the datacenters for the latency sensitivity
                datacenters = self.get_datacenters_for_latency(latency)
                ## loop over the datacenters, check the remained slots for each datacenter
                ## and check the cost when add the server to this datacenter
                remained_slots = []
                costs = []
                for datacenter in datacenters:
                    ## get the remained slots
                    remained_slot = self.resource.calculate_remained_slots_for_datacenter(current_time, datacenter)
                    ## get the cost
                    cost = self.calculate_cost_for_datacenter(current_time, datacenter)
                    ## append the remained slots and the cost
                    remained_slots.append(remained_slot)
                    costs.append(cost)
                ## now we sorted the datacenters in cost increasing order
                sorted_cost_idx = np.argsort(costs)
                ## loop over the datacenters
                for idx in sorted_cost_idx:
                    ## get the datacenter
                    datacenter = datacenters[idx]
                    ## get the remained slots
                    remained_slot = remained_slots[idx]
                    ## calculate the max number of servers that can be added
                    max_servers = min(demand ,remained_slot)  // server_capacity
                    ## add the servers
                    for _ in range(max_servers):
                        sucess =    self.add_server(current_time, server_generation, datacenter)
                        if sucess:
                            demand -= server_capacity
                        
                    if demand == 0:
                        break


        
    def simulate(self):
        total_objective = 0
        for timestep in self.demand.time_step.unique():
            ## accumulate demand for 10 time steps
            demand =self.get_demand(timestep)
            # sub_demand = self.get_demand([t for t in range(timestep,timestep+20)])
            # pdb.set_trace()
            ## now try with our strategy
            self.strategy(demand, timestep)

            print(f"Time step {timestep}", end=":  ")
            print(f"Profit {self.calculate_profit(timestep, demand):.2f}", end=", ")
            print(f"Objective {self.calculate_objective(timestep, demand):.2f}", end=", ")
            print(f"Cost {self.calculate_cost(timestep):.2f}", end=", ")
            print(f"Revenue {self.calculate_revenue(timestep, demand):.2f}", end=", ")
            print(f"Utilization {self.calculate_utilization(timestep, demand):.2f}", end=", ")
            print(f"Normalized timespan {self.calculate_normalized_timespan(timestep):.2f}")
            print("\n")
            total_objective += self.calculate_objective(timestep, demand)
        print("\n")
        print(f"Total objective {total_objective:2f}")


        
        

   