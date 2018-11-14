# Ryan Jaipersaud, Armaan Thapar
# November 13, 2018
# Convex Optimization

# Retailer Solution

import numpy as np
from scipy.optimize import linprog

# Inequalities Matrix
A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# Results vector for inequalities
b = np.array([60, 60, 60, 0, 0, 0, 0, 0, 0, 30, 30, 30])

# Matrix of equalities
B = np.array([[1, 0, 0, 0, 0, 0, 1, -1, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 1, -1, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 1, -1],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

# Results vector for equalities
d = np.array([10, 50, 90, 1, 1, 1, 0])

# Cost vector
c= [5, 5, 5, 10, 10, 10, 4, 4, 4, 4]

# Calculate Solution
sol = linprog(c, A, b, B, d)

# Print Solution
print(sol)
