# %%
import multiprocessing
import logging
import time
import random
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

import os
import zipfile
from typing import List
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Logging started")


def zip_files(file_paths: List[str], output_zip: str):
    """
    Zip multiple files into a single zip file.

    :param file_paths: List of file paths to be zipped
    :param output_zip: Name of the output zip file
    """
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
                logging.info(f"Added {file} to {output_zip}")
            else:
                logging.info(f"Warning: {file} not found and skipped")

# set file logging
file_handler = logging.FileHandler('logs/solver_v2.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)



# %%

import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from scipy.stats import truncweibull_min
from utils import (load_problem_data,
                   load_solution)

from evaluation import get_actual_demand, evaluation_function

import uuid
import tqdm
import evaluation
import pickle

# %%

def pad_array_to_multiple_of_12(arr):
    # Get the current length of the array
    current_length = len(arr)
    
    # Calculate how many elements we need to add
    elements_to_add = (12 - (current_length % 12)) % 12
    
    # If elements_to_add is 0, it means the array is already divisible by 12
    if elements_to_add == 0:
        return arr
    
    # Create a new array with np.nan padding
    padded_arr = np.pad(arr, (0, elements_to_add), mode='constant', constant_values=np.nan)
    
    return padded_arr

def fill_missing_timestep(df, min_time_step=1, max_time_step=168):
    full_range = pd.DataFrame({'time_step': range(min_time_step, max_time_step + 1)})
    df_filled = pd.merge(full_range, df, on='time_step', how='left')
    numeric_columns = ['high', 'low', 'medium']
    df_filled[numeric_columns] = df_filled[numeric_columns].fillna(0)
    df_filled['server_generation'] = df_filled['server_generation'].ffill()
    df_filled = df_filled.reset_index(drop=True)
    return df_filled

def parse_action_string(action_string):
    server_generation, actions, action_params = action_string.split("|")
    actions = actions.split("-")
    action_params = action_params.split("-")
    action_comb = [parse_action_comb_param(server_generation, action, action_param) for action, action_param in zip(actions, action_params)]
    action_comb = [action for action in action_comb if action is not None]
    return action_comb
    
def parse_action_comb_param(server_generation, action, action_param):
    if action == "buy":
        datacenter_id, average_U = action_param.split(",")
        average_U = float(average_U)
        return {"action": action, "datacenter_id": datacenter_id, "average_U": average_U, "server_generation": server_generation}
    elif action == "dismiss":
        dismiss_age = int(action_param)
        return {"action": action, "dismiss_age": dismiss_age, "server_generation": server_generation}
    elif action == "move":
        datacenter_id, average_U, move_age = action_param.split(",")
        average_U = float(average_U)
        move_age = int(move_age)
        return {"action": action, "datacenter_id": datacenter_id, "average_U": average_U, "move_age": move_age,}
    return None

# %%



class Solver:

    def __init__(self, df_servers, 
                 df_data_centers, 
                 df_selling_prices, 
                 df_price_elasticity_of_demand,
                 time_steps=[1, 168], 
                 verbose=False):
        self.df_servers = df_servers.copy()
        self.df_servers['server_release_time_start'] = self.df_servers['release_time'].apply(lambda x: int(x.strip('[]').split(',')[0]))
        self.df_servers['server_release_time_end'] = self.df_servers['release_time'].apply(lambda x: int(x.strip('[]').split(',')[1]))
        
        self.df_servers_dict = self.df_servers.set_index('server_generation').to_dict('index')

        self.df_selling_prices = df_selling_prices.copy()
        self.df_data_centers = df_data_centers.copy()
        self.df_price_elasticity_of_demand = df_price_elasticity_of_demand.copy()
        self.datacenter_ids_to_index = {datacenter_id: i for i, datacenter_id in enumerate(df_data_centers['datacenter_id'].values)}
        self.df_datacenters_dict = self.df_data_centers.set_index('datacenter_id').to_dict('index')
        self.time_steps = time_steps

        self.verbose = verbose
    
        self.failure_rate = 0.07260491698699582

        self.server_generation_unique = self.df_servers['server_generation'].unique()
        self.server_generation_to_idx = {server_generation: i for i, server_generation in enumerate(self.server_generation_unique)}

        self.sensitivity_unique = ['high', 'medium', 'low']
        self.sensitivity_to_idx = {sensitivity: i for i, sensitivity in enumerate(self.sensitivity_unique)}

        # transform base selling prices to numpy of [num_servers, 1, num_sensitivity_levels]
        self.selling_prices = np.zeros((self.df_servers.shape[0], 1, len(self.sensitivity_unique)))
        for i, row in self.df_selling_prices.iterrows():
            server_idx = self.server_generation_to_idx[row['server_generation']]
            sensitivity_idx = self.sensitivity_to_idx[row['latency_sensitivity']]
            self.selling_prices[server_idx, 0, sensitivity_idx] = row['selling_price']
        
        # trasnform elasticity of demand to numpy of [num_servers, 1, num_sensitivity_levels]
        self.price_elasticity_of_demand = np.zeros((self.df_servers.shape[0], 1, len(self.sensitivity_unique)))
        for i, row in self.df_price_elasticity_of_demand.iterrows():
            server_idx = self.server_generation_to_idx[row['server_generation']]
            sensitivity_idx = self.sensitivity_to_idx[row['latency_sensitivity']]
            self.price_elasticity_of_demand[server_idx, 0, sensitivity_idx] = row['elasticity']

        self.data_center_max_slots = np.zeros((self.df_data_centers.shape[0], self.time_steps[1]))
        for i, row in self.df_data_centers.iterrows():
            datacenter_idx = self.datacenter_ids_to_index[row['datacenter_id']]
            self.data_center_max_slots[datacenter_idx] = row['slots_capacity']

        self.historical_server_ids = [] # FOR ANALYSIS PERPOSES

        self.id_count = 0

    def load_demand(self, demand):
        self.df_demand = demand.copy()
        actual_demand_by_server_generation = {server_generation: demand[demand['server_generation'] == server_generation].sort_values('time_step') 
                                            for server_generation in self.server_generation_unique}
        for key in actual_demand_by_server_generation:
            actual_demand_by_server_generation[key] = fill_missing_timestep(actual_demand_by_server_generation[key])
        self.actual_demand_by_server_generation = np.asarray([actual_demand_by_server_generation[server_generation][['high', 'medium', 'low']] 
                                                              for server_generation in self.server_generation_unique])
        # self.actual_demand_by_server_generation shape is (num_server_generations, num_time_steps, num_sensitivity_levels)

    def generate_random_id(self,):
        return str(uuid.uuid4())
    
    def init_solution(self, demand):
        self.solution = []
        self.load_demand(demand)
        
        # initialize solution objectives
        self.solution_Z = np.zeros((self.df_servers.shape[0], self.time_steps[1], len(self.sensitivity_unique)))
        self.solution_L = np.zeros((self.time_steps[1],))
        self.solution_R = np.zeros((self.time_steps[1],))
        self.solution_C = np.zeros((self.time_steps[1],))
        self.solution_num_servers = np.zeros((self.time_steps[1],))
        self.data_center_slots = np.zeros((self.df_data_centers.shape[0], self.time_steps[1]))
        self.solution_obj = 0
        self.solution_selling_prices = np.ones((self.df_servers.shape[0], self.time_steps[1], len(self.sensitivity_unique))) * self.selling_prices # base selling prices

    def update_demand_by_solution_selling_prices(self,):
        delta_p = (self.solution_selling_prices - self.selling_prices) / self.selling_prices
        delta_d = self.price_elasticity_of_demand * delta_p
        self.actual_demand_by_server_generation *= (1 + delta_d)

    def calculate_new_obj_components(self, cur_num_servers, cur_Z, cur_L, cur_C):
        new_num_servers = self.solution_num_servers + cur_num_servers
        new_Z = self.solution_Z + cur_Z
        new_L = (self.solution_L * self.solution_num_servers + cur_L * cur_num_servers) / np.maximum(new_num_servers, 1)
        new_C = self.solution_C + cur_C
        return new_num_servers, new_Z, new_L, new_C

    def calculate_new_obj(self, new_num_servers, new_Z, new_L, new_C):

        new_Z_failure = (new_Z * (1-self.failure_rate)).astype(int)
        new_Z_failure_demand = np.minimum(new_Z_failure, self.actual_demand_by_server_generation)
        new_U = np.where(new_Z_failure > 0.5,
                            new_Z_failure_demand / np.maximum(1.0, new_Z_failure),
                            np.nan)
        new_U = np.nanmean(new_U, axis=(0, 2))
        new_U = np.nan_to_num(new_U, nan=0)

        new_R = new_Z_failure_demand * self.solution_selling_prices
        new_R = np.nansum(new_R, axis=(0, 2))

        # new_obj = new_U * new_L * (new_R - new_C)
        new_obj = new_R - new_C
        new_obj = np.nansum(new_obj)

        return new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R)
    
    def update_new_obj(self, new_obj, new_num_servers, new_Z, new_L, new_C, new_U, new_R, new_data_center_slots):
        self.solution_obj = new_obj
        self.solution_Z = new_Z
        self.solution_L = new_L
        self.solution_C = new_C
        self.solution_R = new_R
        self.solution_U = new_U
        self.solution_num_servers = new_num_servers
        self.data_center_slots = new_data_center_slots

    def load_solution(self, df_solution, demand):
        self.load_demand(demand)
        self.init_solution()

        server_id_unique = df_solution['server_id'].unique()

        for server_id in tqdm.tqdm(server_id_unique, desc="Loading solution"):
            df_solution_server_id = df_solution.loc[df_solution['server_id'] == server_id]
            cur_num_servers, cur_Z, cur_L, cur_C, cur_data_center_slots = self.calculate_server_id(df_solution_server_id, append_to_solution=True)

            new_num_servers, new_Z, new_L, new_C = self.calculate_new_obj_components(cur_num_servers=cur_num_servers,
                                                                                    cur_Z=cur_Z,
                                                                                    cur_L=cur_L,
                                                                                    cur_C=cur_C)
            new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=new_num_servers,
                                                                                                    new_Z=new_Z,
                                                                                                    new_L=new_L,
                                                                                                    new_C=new_C)
            self.update_new_obj(new_obj=new_obj,
                                new_num_servers=new_num_servers,
                                new_Z=new_Z,
                                new_L=new_L,
                                new_C=new_C,
                                new_U=new_U,
                                new_R=new_R,
                                new_data_center_slots=self.data_center_slots + cur_data_center_slots)

    def calculate_server_id(self, df_solution_server_id, append_to_solution=False):
        # calculate objective functions for a single life cycle of a server
        df_solution_server_id = df_solution_server_id.sort_values('time_step')
        server_id = df_solution_server_id['server_id'].values[0]

        server_generation = df_solution_server_id['server_generation'].values[0] 
        df_cur_server = self.df_servers_dict[server_generation]
        server_idx = self.server_generation_to_idx[server_generation]
        previous_time_step = 0
        previous_datacenter_id = None
        dismiss_age = None

        cur_num_servers = np.zeros(self.solution_num_servers.shape)
        cur_Z = np.zeros(self.solution_Z.shape)
        cur_L = np.zeros((self.time_steps[1],))
        cur_C = np.zeros((self.time_steps[1],))
        cur_data_center_slots = np.zeros(self.data_center_slots.shape)
        
        for i, row in df_solution_server_id.iterrows():
            action = row['action']
            datacenter_id = row['datacenter_id']
            cur_time_step = row['time_step']

            if append_to_solution:
                self.solution.append({
                    "time_step": cur_time_step,
                    "action": action,
                    "datacenter_id": datacenter_id,
                    "server_generation": server_generation,
                    "server_id": server_id
                })
            
            if action == "buy":
                buy_time_step = cur_time_step
                cur_C[buy_time_step - 1] += df_cur_server['purchase_price']

            elif action != "buy":
                previous_datacenter_idx = self.datacenter_ids_to_index[previous_datacenter_id]
                previous_sensitivity = self.df_datacenters_dict[previous_datacenter_id]['latency_sensitivity']
                previous_sensitivity_idx = self.sensitivity_to_idx[previous_sensitivity]
                previous_datacenter_cost_of_energy = self.df_datacenters_dict[previous_datacenter_id]['cost_of_energy']

                cur_data_center_slots[previous_datacenter_idx, previous_time_step - 1:cur_time_step - 1] += df_cur_server['slots_size']
                cur_Z[server_idx, previous_time_step - 1:cur_time_step - 1, previous_sensitivity_idx] += df_cur_server['capacity']
                cur_C[previous_time_step - 1:cur_time_step - 1] += previous_datacenter_cost_of_energy * df_cur_server['energy_consumption']
                if action == "move":
                    cur_C[cur_time_step - 1] += df_cur_server['cost_of_moving']
                elif action == "dismiss":
                    dismiss_age = cur_time_step - buy_time_step
                
            previous_time_step = cur_time_step
            previous_datacenter_id = datacenter_id

        if dismiss_age is None:
            dismiss_age = min(df_cur_server['life_expectancy'], self.time_steps[1] - buy_time_step + 1)
            cur_time_step = buy_time_step + dismiss_age

            previous_datacenter_idx = self.datacenter_ids_to_index[previous_datacenter_id]
            previous_sensitivity = self.df_datacenters_dict[previous_datacenter_id]['latency_sensitivity']
            previous_sensitivity_idx = self.sensitivity_to_idx[previous_sensitivity]
            previous_datacenter_cost_of_energy = self.df_datacenters_dict[previous_datacenter_id]['cost_of_energy']

            cur_data_center_slots[previous_datacenter_idx, previous_time_step - 1:cur_time_step - 1] += df_cur_server['slots_size']
            cur_Z[server_idx, previous_time_step - 1:cur_time_step - 1, previous_sensitivity_idx] += df_cur_server['capacity']
            cur_C[previous_time_step - 1:cur_time_step - 1] += previous_datacenter_cost_of_energy * df_cur_server['energy_consumption']

        start_idx = buy_time_step - 1
        end_idx = buy_time_step + dismiss_age - 1
        cur_L[start_idx:end_idx] = np.arange(1, dismiss_age + 1) / df_cur_server['life_expectancy']
        cur_C[start_idx:end_idx] += df_cur_server['average_maintenance_fee'] \
            * (1+1.5 * np.arange(1, dismiss_age + 1) / df_cur_server['life_expectancy'] * np.log2(1.5 * np.arange(1, dismiss_age + 1) / df_cur_server['life_expectancy']))

        cur_num_servers[start_idx:end_idx] = 1

        return cur_num_servers, cur_Z, cur_L, cur_C, cur_data_center_slots
                    


    
    
    def remove_nonprofit_server_ids(self, ):
        n_removed = 0
        total_gain = 0
        df_solution = pd.DataFrame(self.solution)
        df_solution_server_id = df_solution[df_solution['action'] == 'buy'].copy()

        for server_id in tqdm.tqdm(df_solution_server_id['server_id'].values, desc="Removing non-profit servers"):
            df_solution_server_id_cur = df_solution.loc[df_solution['server_id'] == server_id]
            cur_num_servers, cur_Z, cur_L, cur_C, cur_data_center_slots = self.calculate_server_id(df_solution_server_id_cur, append_to_solution=False)
            
            new_data_center_slots = self.data_center_slots  - cur_data_center_slots
            new_num_servers = self.solution_num_servers - cur_num_servers
            new_Z = self.solution_Z - cur_Z
            new_L = (self.solution_L * self.solution_num_servers - cur_L * cur_num_servers) / np.maximum(new_num_servers, 1)
            new_C = self.solution_C - cur_C

            new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=new_num_servers,
                                                                                                new_Z=new_Z,
                                                                                                new_L=new_L,
                                                                                                new_C=new_C)
            
            if new_obj <= self.solution_obj:
                continue
            
            self.historical_server_ids.append({
                "solution_action": "remove",
                "server_id": server_id,
                "action_string": None,
                "buy_time_step": None,
                "new_solution_obj": new_obj,
                "obj_gain": new_obj - self.solution_obj,
                "merge_with": None,
            })
            
            total_gain += new_obj - self.solution_obj
            n_removed += 1

            self.update_new_obj(new_obj=new_obj,
                                new_num_servers=new_num_servers,
                                new_Z=new_Z,
                                new_L=new_L,
                                new_C=new_C,
                                new_U=new_U,
                                new_R=new_R,
                                new_data_center_slots=new_data_center_slots)

            # df_solution = df_solution.drop(df_solution[df_solution['server_id'] == server_id].index)
            df_solution = df_solution[df_solution['server_id'] != server_id]
            
        self.solution = df_solution.sort_values(["server_id", "time_step",]).to_dict('records')
        logging.info(f"Removed {n_removed} servers with total gain {total_gain}")

    def merge_server_ids(self, merge_gap_sizes=range(10)):
        df_solution = pd.DataFrame(self.solution)

        total_gain = 0
        n_merged = 0

        # merging
        for gap_size in tqdm.tqdm(merge_gap_sizes, desc="Merging servers"):
            for server_generation in self.server_generation_unique:
                while True:
                    df_solution_server = df_solution.loc[df_solution['server_generation'] == server_generation]
                    df_solution_server_id = df_solution_server[df_solution_server['action'] == 'buy']
                    df_solution_server_id = df_solution_server_id.set_index('server_id').join( \
                        df_solution_server[df_solution_server['action'] == 'move'][['server_id', 'time_step']].set_index('server_id'), \
                        rsuffix='_move')
                    df_solution_server_id = df_solution_server_id.join( \
                        df_solution_server[df_solution_server['action'] == 'dismiss'][['server_id', 'time_step']].set_index('server_id'), \
                        rsuffix='_dismiss')

                    # if time_step_dismiss is None then time_step_dismiss = time_step + 96
                    df_solution_server_id['time_step_dismiss'] = df_solution_server_id['time_step_dismiss'].fillna(df_solution_server_id['time_step'] + 96)
                    df_solution_server_id['age'] = df_solution_server_id['time_step_dismiss'] - df_solution_server_id['time_step']
                    df_solution_server_id_with_dismiss = df_solution_server_id[df_solution_server_id['age'] < 96].reset_index(drop=False)

                    # see if any server is dismissed before other servers are bought, every time_step_dismiss should be greater than time_step
                    cross_diff = df_solution_server_id_with_dismiss['time_step'].values - df_solution_server_id_with_dismiss['time_step_dismiss'].values.reshape(-1, 1)

                    indexes_1, indexes_2 = np.where(np.abs(cross_diff) <= gap_size)
                    # indexes_1, indexes_2 = np.where(np.logical_and(cross_diff >= -gap_size, cross_diff <=  0))
                    # indexes_1, indexes_2 = np.where(np.logical_and(cross_diff <= gap_size, cross_diff >=  0))
                    removed_indexes_2 = set()
                    changed = False
                    for idx_1 in np.unique(indexes_1):
                        idx_1 = idx_1.item()
                        if idx_1 in removed_indexes_2:
                            continue

                        cur_age = df_solution_server_id_with_dismiss['age'].values[idx_1]
                        cur_indexes_2 = indexes_2[indexes_1 == idx_1]

                        # remove removed_indexes_2 from cur_indexes_2
                        cur_indexes_2 = np.array([idx for idx in cur_indexes_2 if idx not in removed_indexes_2 and idx != idx_1])
                        if len(cur_indexes_2) == 0:
                            continue

                        # check age
                        cur_age_2 = df_solution_server_id_with_dismiss['age'].values[cur_indexes_2]
                        cross_diff_2 = cross_diff[idx_1, cur_indexes_2]
                        cur_indexes_2 = cur_indexes_2[cur_age + cur_age_2 + cross_diff_2 <= 96]
                        if len(cur_indexes_2) == 0:
                            continue

                        cur_indexes_2_argsort = np.argsort(df_solution_server_id_with_dismiss['age'].values[cur_indexes_2])[::-1]
                        cur_indexes_2 = cur_indexes_2[cur_indexes_2_argsort]
                        for idx_2 in cur_indexes_2[:1]:

                            server_id_1 = df_solution_server_id_with_dismiss['server_id'].values[idx_1]
                            server_id_2 = df_solution_server_id_with_dismiss['server_id'].values[idx_2]

                            df_solution_server_id_1 = df_solution_server.loc[df_solution_server['server_id'] 
                                                                                             == server_id_1].sort_values('time_step')
                            df_solution_server_id_2 = df_solution_server.loc[df_solution_server['server_id'] 
                                                                                             == server_id_2].sort_values('time_step')

                            
                            # remove calculations for server_id_1 and server_id_2
                            cur_num_servers_1, cur_Z_1, cur_L_1, cur_C_1, cur_data_center_slots_1 = self.calculate_server_id(df_solution_server_id_1, append_to_solution=False)
                            cur_num_servers_2, cur_Z_2, cur_L_2, cur_C_2, cur_data_center_slots_2 = self.calculate_server_id(df_solution_server_id_2, append_to_solution=False)

                            df_solution_new_server_id = pd.concat([df_solution_server_id_1, df_solution_server_id_2]).sort_values('time_step')

                            # drop dismiss of the server_id_1
                            df_solution_new_server_id = df_solution_new_server_id[np.logical_not(np.logical_and(df_solution_new_server_id['action'] == 'dismiss',
                                                                                                                df_solution_new_server_id['server_id'] == server_id_1))]
                            
                            # change buy of server_id_2 to move
                            df_solution_new_server_id.loc[np.logical_and(df_solution_new_server_id['action'] == 'buy',
                                                                        df_solution_new_server_id['server_id'] == server_id_2), 'action'] = 'move'
                            
                            df_solution_new_server_id['server_id'] = server_id_1

                            cur_num_servers_new, cur_Z_new, cur_L_new, cur_C_new, cur_data_center_slots_new = self.calculate_server_id(df_solution_new_server_id, append_to_solution=False)

                            new_data_center_slots = self.data_center_slots + cur_data_center_slots_new - cur_data_center_slots_1 - cur_data_center_slots_2 
                            if np.any(self.data_center_max_slots < new_data_center_slots):
                                continue

                            new_num_servers = self.solution_num_servers + cur_num_servers_new - cur_num_servers_1 - cur_num_servers_2
                            new_Z = self.solution_Z + cur_Z_new - cur_Z_1 - cur_Z_2
                            new_L = (self.solution_L * self.solution_num_servers + cur_L_new * cur_num_servers_new - cur_L_1 * cur_num_servers_1 - cur_L_2 * cur_num_servers_2) / np.maximum(new_num_servers, 1)
                            new_C = self.solution_C + cur_C_new - cur_C_1 - cur_C_2

                            new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=new_num_servers,
                                                                                                                new_Z=new_Z,
                                                                                                                new_L=new_L,
                                                                                                                new_C=new_C)

                            if new_obj <= self.solution_obj:
                                continue

                            total_gain += new_obj - self.solution_obj
                            n_merged += 1
                            
                            self.update_new_obj(new_obj=new_obj,
                                                new_num_servers=new_num_servers,
                                                new_Z=new_Z,
                                                new_L=new_L,
                                                new_C=new_C,
                                                new_U=new_U,
                                                new_R=new_R,
                                                new_data_center_slots=new_data_center_slots)
                            
                            df_solution = df_solution[np.logical_not(np.logical_or(df_solution['server_id'] == server_id_1, 
                                                                                   df_solution['server_id'] == server_id_2))]
                            df_solution = pd.concat([df_solution, df_solution_new_server_id])

                            self.historical_server_ids.append({
                                "solution_action": "merge",
                                "server_id": server_id_1,
                                "action_string": None,
                                "buy_time_step": None,
                                "new_solution_obj": new_obj,
                                "obj_gain": new_obj - self.solution_obj,
                                "merge_with": server_id_2,
                            })
                            
                            removed_indexes_2.add(idx_1)
                            removed_indexes_2.add(idx_2)
                            changed = True
                            break
                    if not changed:
                        break
        logging.info(f"Merged {n_merged} servers with total gain {total_gain}")
        
        self.solution = df_solution.sort_values(["server_id", "time_step",]).to_dict('records')


    def search_servers(self, df_single_server):
        self.df_single_server = df_single_server

        current_n_buys = len([action for action in self.solution if action['action'] == 'buy'])
        cur_obj = self.solution_obj

        for action_string in tqdm.tqdm(self.df_single_server['action_string'].values, desc="Searching actions"):
            if "|buy-dismiss|" in action_string:
                self.search_buy_dismiss_combination(action_string)
            elif "|buy-move-dismiss|" in action_string:
                self.search_buy_move_dismiss_combination(action_string)

        new_n_buys = len([action for action in self.solution if action['action'] == 'buy'])
        new_obj = self.solution_obj

        logging.info(f"Added {new_n_buys - current_n_buys} servers with total gain {new_obj - cur_obj}")

    def search_buy_move_dismiss_combination(self, action_string):
        action_comb = parse_action_string(action_string)

        server_generation = action_comb[0]['server_generation']

        datacenter_id_1 = action_comb[0]['datacenter_id']
        datacenter_id_2 = action_comb[1]['datacenter_id']

        move_age = action_comb[1]['move_age']
        dissmiss_age = action_comb[2]['dismiss_age']


        df_cur_server = self.df_servers_dict[server_generation]
        df_datacenter_1 = self.df_datacenters_dict[datacenter_id_1]
        df_datacenter_2 = self.df_datacenters_dict[datacenter_id_2]

        utilization_threshold_1 = action_comb[0]['average_U']
        utilization_threshold_2 = action_comb[1]['average_U']

        sensitivity_1 = df_datacenter_1['latency_sensitivity']
        sensitivity_2 = df_datacenter_2['latency_sensitivity']

        datacenter_cost_of_energy_1 = df_datacenter_1['cost_of_energy']
        datacenter_cost_of_energy_2 = df_datacenter_2['cost_of_energy']

        server_capacity = df_cur_server['capacity']
        server_energy_consumption = df_cur_server['energy_consumption']
        server_life_expectancy = df_cur_server['life_expectancy']
        server_average_maintenance_fee = df_cur_server['average_maintenance_fee']
        server_purchase_price = df_cur_server['purchase_price']
        server_cost_of_moving = df_cur_server['cost_of_moving']
        server_slots_size = df_cur_server['slots_size']
        server_release_time_start = df_cur_server['server_release_time_start']
        server_release_time_end = df_cur_server['server_release_time_end']

        datacenter_idx_1 = self.datacenter_ids_to_index[datacenter_id_1]
        datacenter_idx_2 = self.datacenter_ids_to_index[datacenter_id_2]
        server_idx = self.server_generation_to_idx[server_generation]
        sensitivity_idx_1 = self.sensitivity_to_idx[sensitivity_1]
        sensitivity_idx_2 = self.sensitivity_to_idx[sensitivity_2]

        demand_arr_1 = self.actual_demand_by_server_generation[server_idx, :, self.sensitivity_to_idx[sensitivity_1]]
        demand_arr_2 = self.actual_demand_by_server_generation[server_idx, :, self.sensitivity_to_idx[sensitivity_2]]
        total_buys = 0
        for buy_time_step in range(server_release_time_start, server_release_time_end + 1):
            start_idx_1 = buy_time_step - 1
            end_idx_1 = buy_time_step + move_age - 1
            start_idx_2 = buy_time_step + move_age - 1
            end_idx_2 = buy_time_step + dissmiss_age - 1

            demand_subarr_1 = demand_arr_1[start_idx_1:end_idx_1]
            demand_subarr_2 = demand_arr_2[start_idx_2:end_idx_2]
            if demand_subarr_1.shape[0] < move_age or demand_subarr_1.shape[0] + demand_subarr_2.shape[0] < dissmiss_age:
                break

            while True:

                # check utilization condition for datacenter 1
                demand_subarr_1_capped = np.minimum(demand_subarr_1, server_capacity)
                # if demand_subarr_1_capped[0] / server_capacity < utilization_threshold_1: # check the first time step
                #     break

                demand_subarr_1_capped = pad_array_to_multiple_of_12(demand_subarr_1_capped)
                demand_subarr_1_capped = demand_subarr_1_capped.reshape(-1, 12)
                average_utilization_1 = np.nanmean(demand_subarr_1_capped / server_capacity, axis=1)
                if np.any(average_utilization_1 < utilization_threshold_1):
                    break

                # check utilization condition for datacenter 2
                demand_subarr_2_capped = np.minimum(demand_subarr_2, server_capacity)
                demand_subarr_2_capped = pad_array_to_multiple_of_12(demand_subarr_2_capped)
                demand_subarr_2_capped = demand_subarr_2_capped.reshape(-1, 12)
                average_utilization_2 = np.nanmean(demand_subarr_2_capped / server_capacity, axis=1)
                if np.any(average_utilization_2 < utilization_threshold_2):
                    break

                # check data center slots
                new_data_center_slots = np.copy(self.data_center_slots)
                new_data_center_slots[datacenter_idx_1, start_idx_1:end_idx_1] += server_slots_size
                new_data_center_slots[datacenter_idx_2, start_idx_2:end_idx_2] += server_slots_size
                if np.any(self.data_center_max_slots < new_data_center_slots):
                    break

                # check if adding the server is beneficial by calculating the objective function

                ### calculate for the new server
                cur_Z = np.zeros(self.solution_Z.shape)
                cur_Z[server_idx, start_idx_1:end_idx_1, sensitivity_idx_1] = server_capacity
                cur_Z[server_idx, start_idx_2:end_idx_2, sensitivity_idx_2] = server_capacity

                cur_L = np.zeros(self.solution_L.shape)
                cur_L[start_idx_1:end_idx_2] = np.arange(1, dissmiss_age + 1) / server_life_expectancy

                cur_E = np.zeros(self.solution_L.shape)
                cur_E[start_idx_1:end_idx_2] = datacenter_cost_of_energy_1 * server_energy_consumption
                cur_E[start_idx_2:end_idx_2] = datacenter_cost_of_energy_2 * server_energy_consumption

                cur_alpha = np.zeros(self.solution_L.shape)
                cur_alpha[start_idx_1:end_idx_2] = server_average_maintenance_fee \
                    * (1+1.5 * np.arange(1, dissmiss_age + 1) / server_life_expectancy * np.log2(1.5 * np.arange(1, dissmiss_age + 1) / server_life_expectancy))
                cur_C = cur_E + cur_alpha
                cur_C[start_idx_1] += server_purchase_price
                cur_C[start_idx_2] += server_cost_of_moving

                cur_num_servers = np.zeros(self.solution_L.shape)
                cur_num_servers[start_idx_1:end_idx_2] = 1

                new_num_servers, new_Z, new_L, new_C = self.calculate_new_obj_components(cur_num_servers=cur_num_servers,
                                                                                        cur_Z=cur_Z,
                                                                                        cur_L=cur_L,
                                                                                        cur_C=cur_C)
                new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=new_num_servers,
                                                                                                    new_Z=new_Z,
                                                                                                    new_L=new_L,
                                                                                                    new_C=new_C)
                if new_obj - self.solution_obj <= 0:
                    break
                new_server_id = self.generate_random_id()
                
                self.historical_server_ids.append({
                    "solution_action": "add",
                    "server_id": new_server_id,
                    "action_string": action_string,
                    "buy_time_step": buy_time_step,
                    "new_solution_obj": new_obj,
                    "obj_gain": new_obj - self.solution_obj,
                    "merge_with": None,
                })

                self.update_new_obj(new_obj=new_obj,
                                    new_num_servers=new_num_servers,
                                    new_Z=new_Z,
                                    new_L=new_L,
                                    new_C=new_C,
                                    new_U=new_U,
                                    new_R=new_R,
                                    new_data_center_slots=new_data_center_slots)

                self.add_buy_move_dismiss_action(datacenter_id_1=datacenter_id_1,
                                                    datacenter_id_2=datacenter_id_2,
                                                    server_generation=server_generation,
                                                    move_age=move_age,
                                                    dismiss_age=dissmiss_age,
                                                    time_step=buy_time_step,
                                                    server_id=new_server_id)
                total_buys += 1
        # if self.verbose and total_buys > 0:
        #     print(f"Bought {total_buys} servers with action string {action_string}")

        
    def search_buy_dismiss_combination(self, action_string, ):
        action_comb = parse_action_string(action_string)

        server_generation = action_comb[0]['server_generation']
        datacenter_id = action_comb[0]['datacenter_id']

        df_cur_server = self.df_servers_dict[server_generation]
        df_datacenter = self.df_datacenters_dict[datacenter_id]

        utilization_threshold = action_comb[0]['average_U']

        sensitivity = df_datacenter['latency_sensitivity']
        datacenter_cost_of_energy = df_datacenter['cost_of_energy']

        server_capacity = df_cur_server['capacity']
        server_energy_consumption = df_cur_server['energy_consumption']
        server_life_expectancy = df_cur_server['life_expectancy']
        server_average_maintenance_fee = df_cur_server['average_maintenance_fee']
        server_purchase_price = df_cur_server['purchase_price']
        server_slots_size = df_cur_server['slots_size']
        server_release_time_start = df_cur_server['server_release_time_start']
        server_release_time_end = df_cur_server['server_release_time_end']
        dissmiss_age = action_comb[1]['dismiss_age']

        datacenter_idx = self.datacenter_ids_to_index[datacenter_id]
        server_idx = self.server_generation_to_idx[server_generation]
        sensitivity_idx = self.sensitivity_to_idx[sensitivity]
        demand_arr = self.actual_demand_by_server_generation[server_idx, :, sensitivity_idx]
        total_buys = 0
        for buy_time_step in range(server_release_time_start, server_release_time_end + 1):
            start_idx = buy_time_step - 1
            end_idx = buy_time_step + dissmiss_age - 1
            demand_subarr = demand_arr[start_idx:end_idx]
            if demand_subarr.shape[0] < dissmiss_age:
                break

            while True:
                # check utilization condition
                demand_subarr_capped = np.minimum(demand_subarr, server_capacity)

                # if demand_subarr_capped[0] / server_capacity < utilization_threshold: # check the first time step
                #     break
                
                demand_subarr_capped = pad_array_to_multiple_of_12(demand_subarr_capped)
                demand_subarr_capped = demand_subarr_capped.reshape(-1, 12)
                average_utilization = np.nanmean(demand_subarr_capped / server_capacity, axis=1)
                if np.any(average_utilization < utilization_threshold):
                    break

                # check data center slots
                new_data_center_slots = np.copy(self.data_center_slots)
                new_data_center_slots[datacenter_idx, start_idx:end_idx] += server_slots_size

                if np.any(self.data_center_max_slots < new_data_center_slots):
                    break
                
                # check if adding the server is beneficial by calculating the objective function
                cur_Z = np.zeros(self.solution_Z.shape)
                cur_Z[server_idx, start_idx:end_idx, sensitivity_idx] = server_capacity
                
                cur_L = np.zeros((self.time_steps[1],))
                cur_L[start_idx:end_idx] = np.arange(1, dissmiss_age + 1) / server_life_expectancy

                cur_E = np.zeros((self.time_steps[1],))
                cur_E[start_idx:end_idx] = datacenter_cost_of_energy * server_energy_consumption

                cur_alpha = np.zeros((self.time_steps[1],))
                cur_alpha[start_idx:end_idx] = server_average_maintenance_fee \
                    * (1+1.5 * np.arange(1, dissmiss_age + 1) / server_life_expectancy * np.log2(1.5 * np.arange(1, dissmiss_age + 1) / server_life_expectancy))
                cur_C = cur_E + cur_alpha
                cur_C[start_idx] += server_purchase_price

                cur_num_servers = np.zeros((self.time_steps[1],))
                cur_num_servers[start_idx:end_idx] = 1


                new_num_servers, new_Z, new_L, new_C = self.calculate_new_obj_components(cur_num_servers=cur_num_servers,
                                                                                        cur_Z=cur_Z,
                                                                                        cur_L=cur_L,
                                                                                        cur_C=cur_C)
                new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=new_num_servers,
                                                                                                       new_Z=new_Z,
                                                                                                       new_L=new_L,
                                                                                                       new_C=new_C)
                
                if new_obj - self.solution_obj <= 0:
                    break

                new_server_id = self.generate_random_id()
                
                self.historical_server_ids.append({
                    "solution_action": "add",
                    "server_id": new_server_id,
                    "action_string": action_string,
                    "buy_time_step": buy_time_step,
                    "new_solution_obj": new_obj,
                    "obj_gain": new_obj - self.solution_obj,
                    "merge_with": None,
                })
                self.update_new_obj(new_obj=new_obj,
                                    new_num_servers=new_num_servers,
                                    new_Z=new_Z,
                                    new_L=new_L,
                                    new_C=new_C,
                                    new_U=new_U,
                                    new_R=new_R,
                                    new_data_center_slots=new_data_center_slots)
                
                self.add_buy_dismiss_action(datacenter_id=datacenter_id, 
                                        server_generation=server_generation, 
                                        dismiss_age=dissmiss_age, 
                                        time_step=buy_time_step,
                                        server_id=new_server_id)
                total_buys += 1
                    
        # if self.verbose and total_buys > 0:
        #     logging.info(f"Bought {total_buys} servers with action string {action_string}")

    def add_buy_move_dismiss_action(self, datacenter_id_1, datacenter_id_2, server_generation, move_age, dismiss_age, time_step, server_id=None):
        server_life_expectancy = self.df_servers_dict[server_generation]['life_expectancy']
        if server_id is None:
            server_id = self.generate_random_id()
        self.solution.append({
            "time_step": time_step,
            "action": "buy",
            "datacenter_id": datacenter_id_1,
            "server_generation": server_generation,
            "server_id": server_id
        })

        if time_step + move_age <= self.time_steps[1]:
            self.solution.append({
                "time_step": time_step + move_age,
                "action": "move",
                "datacenter_id": datacenter_id_2,
                "server_generation": server_generation,
                "server_id": server_id
            })

        if dismiss_age < server_life_expectancy and time_step + dismiss_age <= self.time_steps[1]:
            self.solution.append({
                "time_step": time_step + dismiss_age,
                "action": "dismiss",
                "datacenter_id": datacenter_id_2,
                "server_generation": server_generation,
                "server_id": server_id
            })

    def add_buy_dismiss_action(self, datacenter_id, server_generation, dismiss_age, time_step, server_id=None):
        server_life_expectancy = self.df_servers_dict[server_generation]['life_expectancy']
        
        if server_id is None:
            server_id = self.generate_random_id()
        self.solution.append({
            "time_step": time_step,
            "action": "buy",
            "datacenter_id": datacenter_id,
            "server_generation": server_generation,
            "server_id": server_id
        })
        if dismiss_age < server_life_expectancy and time_step + dismiss_age <= self.time_steps[1]:
            self.solution.append({
                "time_step": time_step + dismiss_age,
                "action": "dismiss",
                "datacenter_id": datacenter_id,
                "server_generation": server_generation,
                "server_id": server_id
            })

    def save_checkpoint(self, path):
        data = {
            "solution": self.solution,
            "solution_num_servers": self.solution_num_servers,
            "solution_Z": self.solution_Z,
            "solution_L": self.solution_L,
            "solution_C": self.solution_C,
            "datacenter_slots": self.data_center_slots,
            "historical_server_ids": self.historical_server_ids,
            "solution_selling_prices": self.solution_selling_prices,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_checkpoint(self, path, demand):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.solution = data['solution']
            cur_num_servers = data['solution_num_servers']
            cur_Z = data['solution_Z']
            cur_L = data['solution_L']
            cur_C = data['solution_C']
            cur_data_center_slots = data['datacenter_slots']
            self.solution_selling_prices = data['solution_selling_prices']
            if 'historical_server_ids' in data:
                self.historical_server_ids = data['historical_server_ids']

            self.load_demand(demand)

            new_obj, (new_num_servers, new_Z, new_L, new_C, new_U, new_R) = self.calculate_new_obj(new_num_servers=cur_num_servers,
                                                                                                new_Z=cur_Z,
                                                                                                new_L=cur_L,
                                                                                                new_C=cur_C)
            self.update_new_obj(new_obj=new_obj,
                                new_num_servers=new_num_servers,
                                new_Z=new_Z,
                                new_L=new_L,
                                new_C=new_C,
                                new_U=new_U,
                                new_R=new_R,
                                new_data_center_slots=cur_data_center_slots)
    def solution_to_dataframe(self,):
        df_fleet = pd.DataFrame(self.solution)
        df_pricing_strategy = []
        for time_step in range(1, self.time_steps[1]+1):
            for server_generation in self.server_generation_to_idx.keys():
                for sensitivity in self.sensitivity_to_idx.keys():
                    df_pricing_strategy.append({
                        "time_step": time_step,
                        "server_generation": server_generation,
                        "latency_sensitivity": sensitivity,
                        "price": self.solution_selling_prices[self.server_generation_to_idx[server_generation], 
                                                              time_step-1, 
                                                              self.sensitivity_to_idx[sensitivity]]})
        df_pricing_strategy = pd.DataFrame(df_pricing_strategy)
        return df_fleet, df_pricing_strategy
                    
            



# %%
from copy import deepcopy

def solve_demand(demand, 
                 df_single_server, 
                 verbose=False, 
                 failure_rate_r=1.0, 
                 df_solution=None, 
                 output_file=None, 
                 restore_checkpoint_path=None, 
                 checkpoint_path=None, 
                 n_solving_loops=5):
    _, datacenters, servers, selling_prices, elasticity = load_problem_data()
    solver = Solver(df_servers=servers, 
                        df_data_centers=datacenters, 
                        df_selling_prices=selling_prices,
                        df_price_elasticity_of_demand=elasticity,
                        verbose=verbose)
    solver.failure_rate = 0.07260491698699582 * failure_rate_r
    if restore_checkpoint_path is not None and os.path.exists(restore_checkpoint_path):
        solver.load_checkpoint(restore_checkpoint_path, demand=demand)
    elif df_solution is not None:
        solver.load_solution(df_solution, demand=demand)
    else:
        solver.init_solution(demand=demand)

    best_obj = solver.solution_obj
    logging.info(f"Initial solution: {solver.solution_obj}")

    
    def save_checkpoint():
        if output_file is not None:
            df_fleet, df_pricing_strategy = solver.solution_to_dataframe()    
            save_solution(fleet=df_fleet,
                          pricing_strategy=df_pricing_strategy,
                          path=output_file)
        if checkpoint_path is not None:
            solver.save_checkpoint(checkpoint_path)

    solver.remove_nonprofit_server_ids()
    if solver.solution_obj > best_obj:
        logging.info(f"New best solution: {solver.solution_obj}")
        best_obj = solver.solution_obj
        save_checkpoint()
            
    for _ in range(n_solving_loops):
        for step in range(3):
            if step == 0:
                solver.search_servers(df_single_server=df_single_server)
            elif step == 1:
                solver.merge_server_ids(merge_gap_sizes=range(12))
            elif step == 2:
                solver.remove_nonprofit_server_ids()

            if solver.solution_obj > best_obj:
                logging.info(f"New best solution: {solver.solution_obj}")
                best_obj = solver.solution_obj
                save_checkpoint()
    return solver

# %% [markdown]
# # solve seeds

# %%
def solve_seed(seed):
    OUTPUT_FOLDER = "solution_v2"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    output_file = f"{OUTPUT_FOLDER}/{seed}.json"
    checkpoint_path = f"{OUTPUT_FOLDER}/{seed}.pkl"
        
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

    np.random.seed(seed)
    actual_demand = evaluation.get_actual_demand(demand)

    df_single_server_buydismiss = pd.read_csv("resources/df_single_server_results_buydismiss_P.csv")
    df_single_server_buymovedismiss = pd.read_csv("resources/df_single_server_results_buymovedismiss_P.csv")

    list_df_single_server_buymovedismiss_filtered = []
    for server_generataion in df_single_server_buymovedismiss['server_generation'].unique():
        df_server_generation = df_single_server_buymovedismiss[df_single_server_buymovedismiss['server_generation'] == server_generataion]
        df_server_generation = df_server_generation.iloc[:int(len(df_server_generation) * 0.2)]
        list_df_single_server_buymovedismiss_filtered.append(df_server_generation)
    df_single_server_buymovedismiss_filtered = pd.concat(list_df_single_server_buymovedismiss_filtered)
    df_single_server_buymovedismiss_filtered = df_single_server_buymovedismiss_filtered.sort_values('score_per_slot', ascending=False).reset_index(drop=True)

    df_single_server = pd.concat([df_single_server_buydismiss, df_single_server_buymovedismiss_filtered])

    solver = solve_demand(demand=actual_demand,
                            df_single_server=df_single_server,
                            verbose=True,
                            failure_rate_r=1.0,
                            n_solving_loops=50,
                            output_file=output_file,
                            restore_checkpoint_path="resources/base_solution.pkl",
                            checkpoint_path=checkpoint_path)
    
    return solver


if __name__ == "__main__":
    

    seeds =  [2381, 5351, 6047, 6829, 9221, 9859, 8053, 1097, 8677, 2521]
    
    processes = []
    for seed in seeds:
        p = multiprocessing.Process(target=solve_seed, 
                                    args=(seed,), 
                                    name=f"Solving Seed-{seed}")
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()



