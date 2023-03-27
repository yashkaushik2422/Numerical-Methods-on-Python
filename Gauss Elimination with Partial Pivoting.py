#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np

def gaussEliminationPP(a_matrix,b_matrix):
    
    #Adding contigencies to prevent problems
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("Error: matrix not square")
        return
    
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("Error: Constant vector is incorrectly sized")
        return
    
    #intitalize necessary variables
    n = len(b_matrix)
    m = n - 1
    i = 0
    x = np.zeros(n)
    new_line = "\n"
    
    #create augmented matrix through concatenate function
    augmented_matrix = np.concatenate((a_matrix,b_matrix), axis=1, dtype=float)
    print(f"The initial augmented matrix is:{new_line},{augmented_matrix}")
    print("Solving for the upper-traingle matrix:")
    
    # Applying Gaussian Eliminatio w/ Pivoting
    while i < n:
        
        # Partial Pivoting
        for p in range(i+1,n):
            if abs(augmented_matrix[i,i]) < abs(augmented_matrix[p,i]):
                augmented_matrix[[p,i]] = augmented_matrix[[i,p]]
                
        if augmented_matrix[i,i] == 0.0:
            print("Divide by zero error!")
            return
        
        for j in range(i+1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            print(augmented_matrix) #to see each step of elimination 
        
        i = i + 1
   
     #Backward substitution to solve for x-matrix:
    x[m]= augmented_matrix[m][n] / augmented_matrix[m][n]
    
    for k in range(n - 2, -1, -1):
        x[k] = augmented_matrix[k][n]
        
        for j in range(k + 1, n):
            x[k] = x[k] - augmented_matrix[k][j] * x[j]
            
        x[k] = x[k] / augmented_matrix[k][k] 
        
    # Displaying solution
    print(f"The following x-vector matrix solves the above equations:")
    for answer in range(n):
        print(f"x{answer} is {x[answer]}")
        
v_matrix= np.array([[3,3,1,5],
[2,2,1,1],
[2,1,0,1],
[9,3,2,1]])
c_matrix= np.array([[6],
[2],
[0],
[-3]])

gaussEliminationPP(v_matrix,c_matrix)
        


# In[16]:


#Check

A = np.array([[3,3,1,5],
[2,2,1,1],
[2,1,0,1],
[9,3,2,1]])

B = np.array([[6],
[2],
[0],
[-3]])

A_inv = np.linalg.inv(A)

x = A_inv.dot(B)

print(x)


# In[ ]:




