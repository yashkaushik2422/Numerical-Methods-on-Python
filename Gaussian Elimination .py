#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np

def gauss_elimination(a_matrix,b_matrix):

#add contingencies
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("Error:Square matrix not given")
        return
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("Error: Constant vector incorrectly sized")
        return
    
    print("matrix is acceptable")
    
    #Initialization of necessary variables
    n = len(b_matrix)
    m = n - 1
    i = 0
    j = i - 1
    x = np.zeros(n)
    new_line = "\n"
    
    # Create augmented matrix that concatenates b to a
    augmented_matrix = np.concatenate((a_matrix, b_matrix), axis=1, dtype=float)
    print(f"The initial aug matrix is: {new_line} {augmented_matrix}")
    print("Solving for the upper traingular matrix:")
    
    # Applying Gauss Elimination
    
    while i < n:
        if augmented_matrix[i][i] == 0.0: #fail-safe to avoid 0 in diagonals
            print("Divide by zero error")
            return
        
        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            print(augmented_matrix) #visualizing intermediate steps
            
        i = i + 1    
        
    # Backwards substitution 
    x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]
    
    for k in range(n-2,-1,-1):
        x[k] = augmented_matrix[k][n]
        
        for j in range(k + 1, n):
            x[k] = x[k] / augmented_matrix[k][k]
            
        #Displaying Solutions
        print(f"The following x-vector matrix solves the above matrix:")
        for answer in range(n):
            print(f"x{answer} is {x[answer]}")
        
    
v_matrix= np.array([[1,1,3],[0,1,3],[-1,3,0]])
c_matrix= np.array([[1],[3],[5]])

gauss_elimination(v_matrix,c_matrix)
    




# In[22]:



A = np.array([[1,1,3],[0,1,3],[-1,3,0]])

B = np.array([[1],[3],[5]])

A_inv = np.linalg.inv(A)

x = A_inv.dot(B)

print(x)


# In[2]:


#Chat GPT code

import numpy as np

def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian Elimination.
    Assumes A is a 3x3 matrix and b is a 3x1 vector.
    Returns the solution vector x.
    """
    
    # Combine A and b into an augmented matrix
    Ab = np.concatenate((A, b), axis=1)
    
    # Perform row operations to put Ab in row echelon form
    for i in range(3):
        # Find the row with the largest absolute value in column i
        max_row = i
        for j in range(i+1, 3):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        
        # Swap the rows so that the largest value is in row i
        Ab[[i,max_row]] = Ab[[max_row,i]]
        
        # Eliminate the values in column i below row i
        for j in range(i+1, 3):
            factor = Ab[j,i] / Ab[i,i]
            Ab[j] -= factor * Ab[i]
    
    # Perform back-substitution to find the solution vector x
    x = np.zeros((3,1))
    x[2] = Ab[2,3] / Ab[2,2]
    x[1] = (Ab[1,3] - Ab[1,2]*x[2]) / Ab[1,1]
    x[0] = (Ab[0,3] - Ab[0,1]*x[1] - Ab[0,2]*x[2]) / Ab[0,0]
    
    return x

A = np.array([[1,1,3],[0,1,3],[-1,3,0]],dtype=np.float64)
b = np.array([[1],[3],[5]],dtype=np.float64)

x = gaussian_elimination(A, b)
print(x)


# In[4]:


import numpy as np

def gauss_elimination(a_matrix,b_matrix):

#add contingencies
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("Error:Square matrix not given")
        return
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("Error: Constant vector incorrectly sized")
        return
    
    print("matrix is acceptable")
    
    #Initialization of necessary variables
    n = len(b_matrix)
    m = n - 1
    i = 0
    j = i - 1
    x = np.zeros(n)
    new_line = "\n"
    
    # Create augmented matrix that concatenates b to a
    augmented_matrix = np.concatenate((a_matrix, b_matrix), axis=1, dtype=float)
    print(f"The initial aug matrix is: {new_line} {augmented_matrix}")
    print("Solving for the upper traingular matrix:")
    
    # Applying Gauss Elimination
    
    while i < n:
        if augmented_matrix[i][i] == 0.0: #fail-safe to avoid 0 in diagonals
            print("Divide by zero error")
            return
        
        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            print(augmented_matrix) #visualizing intermediate steps
            
        i = i + 1    
        
    # Backwards substitution 
    x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]
    
    for k in range(n-2,-1,-1):
        dot_product = np.dot(augmented_matrix[k][k+1:n], x[k+1:n])
        x[k] = (augmented_matrix[k][n] - dot_product) / augmented_matrix[k][k]
            
        #Displaying Solutions
        print(f"The following x-vector matrix solves the above matrix:")
        for answer in range(n):
            print(f"x{answer} is {x[answer]}")
        
    
v_matrix= np.array([[1,1,3],[0,1,3],[-1,3,0]])
c_matrix= np.array([[1],[3],[5]])

gauss_elimination(v_matrix,c_matrix)
    


# In[ ]:




