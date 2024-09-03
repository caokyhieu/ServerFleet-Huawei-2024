from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Number of timesteps
T = 5
max_servers = 100  # Example maximum number of servers

# Step 1: Define variables for buying, dismissing, and lifespan tracking
buy = [model.NewIntVar(0, max_servers, f'buy_{t}') for t in range(T)]
dismiss = [model.NewIntVar(0, max_servers, f'dismiss_{t}') for t in range(T)]
active_servers = [model.NewIntVar(0, max_servers, f'active_servers_{t}') for t in range(T)]

# Lifespan of servers bought at each timestep, tracked at each subsequent timestep
server_lifespan = [[model.NewIntVar(0, T, f'server_lifespan_{i}_{t}') for t in range(T)] for i in range(T)]

# Step 2: Initialize active servers and server lifespan constraints
for t in range(T):
    # The number of active servers is the total bought minus the total dismissed up to that point
    if t == 0:
        model.Add(active_servers[t] == buy[t] - dismiss[t])
    else:
        model.Add(active_servers[t] == active_servers[t - 1] + buy[t] - dismiss[t])

    # Track the lifespan of each server batch
    for i in range(t + 1):
        if i == t:
            # Servers bought at timestep `t` start with a lifespan of 1
            model.Add(server_lifespan[i][t] == 1)
        else:
            # Servers bought at timestep `i` increase in lifespan if still active
            model.Add(server_lifespan[i][t] == server_lifespan[i][t - 1] + 1)

        # Set lifespan to zero if servers bought at `i` are dismissed at `t`
        # FIFO dismissal logic: servers bought at earlier timesteps are dismissed first
        dismissed_upto_t = sum(dismiss[j] for j in range(i + 1, t + 1))
        model.Add(server_lifespan[i][t] == 0).OnlyEnforceIf(dismissed_upto_t > 0)

# Step 3: Create the solver and solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print the results
if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
    print("Buy and Dismiss decisions per timestep:")
    for t in range(T):
        print(f"Timestep {t}: Buy = {solver.Value(buy[t])}, Dismiss = {solver.Value(dismiss[t])}")
        print(f"Active Servers = {solver.Value(active_servers[t])}")
        print("Server Lifespan for this timestep:")
        for i in range(t + 1):
            print(f"  Servers bought at Timestep {i}: Lifespan at Timestep {t} = {solver.Value(server_lifespan[i][t])}")
else:
    print("No solution found.")
