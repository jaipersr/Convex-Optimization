# -*- coding: utf-8 -*-
"""
Ryan Jaipersaid
Convex Optimization
11/18/2018
Minimizing Purchasing Costs
An oil refinery needs to determine how much light,medium and heavy crude to 
purchase in order to meant demand and still minimize on cost. Crude can be 
used to create hydrogen, natural gas, gasoline, jet fuel and heating oil. 
However, each raw material produces different ratios of the product. 
The code below solves for a linear programming optimization problem of the 
form:
    
min c.T*x
s.t Ax >= b
     x > 0
     
The code solves then resolves the LP in standard form:
    
min c.T*x
s.t Ax = b
     x > 0
     
The code solves the dual of the standard LP:
    
min -b.T*lambda
s.t A.T*lambda <=c

The linprog solver from the scipy library is used. 
scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
bounds=None, method='simplex', callback=None, options=None)
"""


import numpy as np
import scipy

# constant vector
c = np.array([55,45,40])

# Inequality matrix Ax >= b
A = np.array([[0.2,0.1,0.1],
             [0.2,0.2,0.1],
             [0.3,0.25,0.2],
             [0.2,0.25,0.28],
             [0.1,0.28,0.32]])

# Upper bound column
     # The division by 10000 is to prevent a floating point error in the solver
b = np.array([[400000],[600000],[1000000],[500000],[300000]])/10000

# linear program solver
# General form of solver shown below
#Minimize:     c^T * x
#
#Subject to:   A_ub * x <= b_ub
#              A_eq * x == b_eq
# the bounds parameter is equivalent to the constant x >= 0

# returns a dictionary
s = scipy.optimize.linprog(c,A_ub = -A,b_ub = -b, bounds = (0, None)) 
print('Question 4 Part C')
print('The LP optimal cost is: $', s['fun']*10000,sep="")
print('The LP optimal argument is: ',s['x']*10000,'\n',)


# This part sets the LP in standard form and solves 
print('Question 4 Part E')
c = np.array([55,45,40,0,0,0,0,0])
I = np.identity(5)

# Ax = b 
A = np.hstack((A[0:5,:],-I)) # stacks an identity matrix to account for slack 
b = np.array([[400000],[600000],[1000000],[500000],[300000]])/10000

# linear program solver
s = scipy.optimize.linprog(c,A_eq = A,b_eq = b, bounds = (0, None)) 

print('The standard LP optimal cost is: $', s['fun']*10000,sep="")
print('The standard LP optimal argument is: ',s['x']*10000)

print('Question 4 Part F')
s = scipy.optimize.linprog(-np.reshape(b,(5)),A_ub = A.T,b_ub = c, bounds = (0, None)) 
# need to multiply by a negative since this is a maximization problem
print('The dual of the standard LP optimal cost is: $', s['fun']*-10000,sep="") 
print('The dual of the standard LP optimal argument is: ',s['x']*10000)
