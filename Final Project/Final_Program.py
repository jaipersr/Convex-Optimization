"""
Ryan Jaipersaud, Armaan Thapar
ChE488 Convex Optimization
12/19/2018

The following code finds the optimal replenishment plan over the course of 5 months
for a small startup retailer and supplier selling t-shirts. The code contains 
3 functions. One for the retailer case, supplier case, and the combined case.

For the independent case:
The code passes the market demand vector to retailer case which returns the the retailer's optimal plan.
The optimal plan for the retailer is then sent to the supplier case since the ordering amount 
of the retailer can be thought of as the demand for the supplier. This will generate 
the replenishment strategy for the supplier. Under the independent case the code 
reduces the BILP to solve 64 LPs. 

For the combined case:
The market demand is sent to the combined case. Under the combined case the code 
reduces the BILP to solve 1024 LPs. The increase in LPs is due to an increase
in variables and constraints that must be solved for to minimize the objective function.

PuLP is used to solved the reduced BILP
"""

import numpy as np
from pulp import *
import itertools
import math
import pandas as pd



# Retailer Solution (Asymmetric Information Exchange)
def Retailer_Case(demand, supplier_demand=False): # supplier_demand = True will print orders as a list for supplier orders

    # Use demand to create indices and y values for brute force solution finding 
    d = demand # Set d to the demand vector given for the number of months
    n = len(d) # number of combinations of ordering plans
    lst = list(map(list,itertools.product([0, 1], repeat=n))) # maps a combination of orders to a list of all combinations of orders
    N = len(d) # Number of months
    indices = [i for i in range(1,N+1)] # List of indices representing months (Used to create variables)

    # Variables
    x_r = LpVariable.dicts("xR",indices, lowBound=0, upBound=115) # Create N variables as a list under x_r (xR_1,..., xR_5)
    s_r = LpVariable.dicts("sR",indices, lowBound=0, upBound=55)  # Create N variables as a list under s_r (sR_1,..., sR_5)

    objective_func = math.inf # Intitialize the minimum objective function to infinity (avoid ignoring large solutions)

    for i in range(len(lst)): 
        y_r = lst[i]    # Take one combination of values for the y vector
        
        prob = LpProblem("Supply Chain Problem: Retailer Case", LpMinimize) # Initialize problem in PuLP, specify that we are minimizing

        term1  = 0  # Want to make single term for objective function, as the second input is automatically taken to be a constraint by PuLP
        for i in range(0,len(indices)):
            term1 += 50*y_r[i]+20*x_r[i+1]+8*s_r[i+1]
        prob += term1, "Objective Function" # Add objective function

        # Add iteratively using the Constraint equations from problem
        for i in range(N):
            if i == 0:
                prob += x_r[i+1]*y_r[i] == d[i] + s_r[i+1], "Constraint " + str(i+1) # Constraint for month 1 (assume no initial storage)
            else:
                prob += s_r[i] + x_r[i+1]*y_r[i] == d[i] + s_r[i+1], "Constraint " + str(i+1)  # Constraints for retailer
                
        prob.solve()  # Solve Problem
        if LpStatus[prob.status] == "Optimal":    # Checks if point is feasible (basic feasible solution)
            
            if value(prob.objective) <  objective_func:   # Checks if value is lower than previous
                objective_func = value(prob.objective)    # Stores value if it is lower
                lowest_soln = np.array([])                # Initialize array to store variables from minimum
                for v in prob.variables():
                    lowest_soln = np.append(lowest_soln, v.varValue)    # Store variables into array

                y_min = y_r                                # Stores y vector at solution as y_r

                data = {'x_R': pd.Series(lowest_soln[N:2*N+1], index = indices), # Store data as dictionary
                        's_R': pd.Series(lowest_soln[0:N], index = indices),
                        'y_R': pd.Series(y_min, index = indices)}

                # Create and format dataframe to show t as title for index 
                df = pd.DataFrame(data)
                df.index.name = ''
                df = df.rename_axis('').rename_axis("t", axis="columns")

    # Prints orders as a vector of demands for the supplier
    print("Retailer Case")
    print("Objective Function:",value(prob.objective))
    print(df)

    # Creates a vector of x_R for Supplier inputs
    if supplier_demand:  
        demands = []
        for v in prob.variables()[N:2*N]:  # indices 0 to 2 are storage, 3 to 6 are orders
            demands.append(v.varValue)
        return demands
    else:
        return objective_func
    
# Supplier Solutions (Asymmetric Information Exchange)
def Supplier_Case(demand):

    # Use demand to create indices and y values for brute force solution finding 
    d = demand # Set d to the demand vector given for the number of months
    n = len(d) # number of combinations of ordering plans
    lst = list(map(list,itertools.product([0, 1], repeat=n))) # maps a combination of orders to a list of all combinations of orders
    N = len(d) # Number of months
    indices = [i for i in range(1,N+1)] # List of indices representing months (Used to create variables)

    # Variables for Supplier
    x_s = LpVariable.dicts("xS",indices, lowBound=0, upBound=110) # Create N variables as a list under x_s (xS_1,..., xS_5)
    s_s = LpVariable.dicts("sS",indices, lowBound=0, upBound=60)  # Create N variables as a list under s_s (sS_1,..., xS_5)

    objective_func = math.inf # Intitialize the minimum objective function to infinity (avoid ignoring large solutions)

    for i in range(len(lst)):
        y_s = lst[i]       # Take one combination of values for the y vector
        
        prob = LpProblem("Supply Chain Problem: Supplier Case", LpMinimize) # Initialize problem in PuLP, specify that we are minimizing

        # Objective Function
        term1  = 0                     # Add as single term as for retailer
        for i in range(len(indices)):
            term1 += 30*y_s[i]+12*x_s[i+1]+11*s_s[i+1]
        prob += term1, "Objective Function"

        # Add constraints
        for i in range(N):
            if i == 0:
                prob += x_s[i+1]*y_s[i] == d[i] + s_s[i+1], "Constraint " + str(i+1)    # Month 1 constraint (assumes 0 initial storage)
            else:
                prob += s_s[i] + x_s[i+1]*y_s[i] == d[i] + s_s[i+1], "Constraint " + str(i+1)  # Months 2 onwards constraints
                
        prob.solve()   #Solves problem
        if LpStatus[prob.status] == "Optimal":     # Checks if point is feasible (basic feasible solution)
            
            if value(prob.objective) <  objective_func:    # Check if value is lower than previous
                objective_func = value(prob.objective)     # Store value if it is lower
                lowest_soln = np.array([])                 # Initialize array to store variables from minimum
                for v in prob.variables():
                    lowest_soln = np.append(lowest_soln, v.varValue)   # Store variables into array
                
                y_min = y_s                                            # Store y vector at minimum

                data = {'x_S': pd.Series(lowest_soln[N:2*N+1], index = indices),  # Store data as dictionary
                        's_S': pd.Series(lowest_soln[0:N], index = indices),
                        'y_S': pd.Series(y_min, index = indices)}

                df = pd.DataFrame(data)  # Create dataframe
                df.index.name = ''
                df = df.rename_axis('').rename_axis("t", axis="columns")  # Add t as title for index


    # Print objective function and dataframe of results             
    print("\nSupplier Case")
    print("Objective Function:",value(prob.objective))
    print(df)

    return objective_func



# Combined Case Solution (Symmetric Information Exchange)
def Combined_Case(demand, order_r_ub = 115, order_s_ub = 110, store_r_ub = 55, store_s_ub = 60): # Kept optional parameters for testing sensitivity

    # Use demand to create indices and y values for brute force solution finding 
    d = demand # Set d to the demand vector given for the number of months
    n = len(d)*2 # number of combinations of ordering plans
    lst = list(map(list,itertools.product([0, 1], repeat=n))) # maps a combination of orders to a list of all combinations of orders
    N = len(d) # Number of months
    indices = [i for i in range(1,N+1)] # List of indices representing months (Used to create variables)

    # Retailer
    x_r = LpVariable.dicts("xR",indices, lowBound=0, upBound=115) # Create N variables as a list under x_r (xR_1,..., xR_5)
    s_r = LpVariable.dicts("sR",indices, lowBound=0, upBound=55)  # Create N variables as a list under s_r (sR_1,..., sR_5)

    objective_func = math.inf # Intitialize the minimum objective function to infinity (avoid ignoring large solutions)

    # Supplier
    x_s = LpVariable.dicts("xS",indices, lowBound=0, upBound=110) # Create N variables as a list under x_s (xS_1,..., xS_5)
    s_s = LpVariable.dicts("sS",indices, lowBound=0, upBound=60)  # Create N variables as a list under s_s (sS_1,..., xS_5)

    for i in range(len(lst)): # iterates over all possible order combinations for y
        y = lst[i] # assigns combination i from lst
        y_r = y[0:N] # takes the first n/2 from combination i and assigns to it retailer ordering plan
        y_s = y[N:int(2*N)] # takes the first n/2 from combination i and assigns to it supplier ordering plan

        prob = LpProblem("Combined Case Problem: Combined Case", LpMinimize) # Initialize problem in PuLP, specify that we are minimizing

        # Objective Function (once again combine into one term to avoid parts being taken as constraints by PuLP)
        term1  = 0
        term2  = 0
        for i in range(0,N):#len(indices)+1):
            term1 += 50*y_r[i]+20*x_r[i+1]+8*s_r[i+1]
            term2 += 30*y_s[i]+12*x_s[i+1]+11*s_s[i+1]
        prob += term1 + term2, "Objective Function" # Assign the objective function

        # Constraints
        for i in range(N):
     
            if i == 0:
                # Constraints for retailer and supplier for month 1
                prob += x_r[i+1]*y_r[i] == d[i] + s_r[i+1], "Retailer Constraint " + str(i+1)
                prob += x_s[i+1]*y_s[i] == x_r[i+1] + s_s[i+1], "Supplier Constraint " + str(i+1)
            else:
                # Constraints for month 2 onwards
                prob += s_r[i] + x_r[i+1]*y_r[i] == d[i] + s_r[i+1], "Retailer Constraint " + str(i+1)
                prob += s_s[i] + x_s[i+1]*y_s[i] ==  x_r[i+1]+ s_s[i+1], "Supplier Constraint " + str(i+1)
                    
        prob.solve() # solve the problem under the constraints
        
        if LpStatus[prob.status] == "Optimal": # Only accept optimal solutions
            if value(prob.objective) <  objective_func: # Only accept solutions that lower the objective function

                objective_func = value(prob.objective) # Define value for objective function to print later on

                lowest_soln = np.array([])  # Initialize array to store variables
                for v in prob.variables():
                    lowest_soln = np.append(lowest_soln, v.varValue) # Store variables
                    
                y_min_r = y_r # Store values for y for both retailer and supplier 
                y_min_s = y_s

                data = {'x_R': pd.Series(lowest_soln[2*N:3*N], index = indices),   # Store data as dictionary
                      'x_S': pd.Series(lowest_soln[3*N:4*N], index = indices),
                      's_R': pd.Series(lowest_soln[0:N], index = indices),
                      's_S': pd.Series(lowest_soln[N:2*N], index = indices),
                        'y_R': pd.Series(y_r, index= indices),
                        'y_S': pd.Series(y_s, index= indices)}
                
                df = pd.DataFrame(data) # conver data to dataframe
                df.index.name = ''
                df= df.rename_axis('').rename_axis("t", axis="columns")

                
                main_prob = prob
    # Print results
    print("\nCombined Case")
    print('Objective Function:',objective_func)
    print(df)
                
    return objective_func


# Main Function
    

d = [54, 103, 116, 105, 141] # Demand for each month
Supplier_Case(Retailer_Case(d, supplier_demand=True)) # Will print results for both retailer and supplier
Combined_Case(d)                                      # Will print results for combined case
