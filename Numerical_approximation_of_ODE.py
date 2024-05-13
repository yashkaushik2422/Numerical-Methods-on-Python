#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
def f(t):
    return np.exp(-0.1 * t) * np.sin(0.75 * t)

def f_prime(t):
    return 0.75 * np.exp(-0.1 * t) * np.cos(0.75 * t) - 0.1 * np.sin(0.75 * t) * np.exp(-0.1 * t)

def f_double_prime(t):
    return -0.15 * np.exp(-0.1 * t) * np.cos(0.75 * t) - 0.5625 * np.exp(-0.1 * t) * np.sin(0.75 * t) + 0.01 * np.exp(-0.1 * t) * np.sin(0.75 * t)

# Parameters
t_start = 0
t_end = 5
num_steps = 50  # dt = 0.1   (change to test different values)
h = (t_end - t_start) / num_steps

# Initialize arrays to store results
t_values = np.linspace(t_start, t_end, num_steps + 1)
analytical_results = f(t_values)
approx_results = np.zeros_like(analytical_results)

# Initialize the initial value
approx_results[0] = f(t_start)

# Implicit backward Euler method
for i in range(1, num_steps + 1):
    approx_results[i] = (approx_results[i - 1] + h * f_prime(t_values[i])) 

# Plot the results
plt.plot(t_values, analytical_results, label="Analytical Solution")
plt.plot(t_values, approx_results, label="Implicit Backward Euler")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.title("Analytical Solution vs Implicit Backward Euler Approximation")
plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function and its first derivative
def f(t):
    return np.exp(-0.1 * t) * np.sin(0.75 * t)

def f_prime(t):
    return 0.75 * np.exp(-0.1 * t) * np.cos(0.75 * t) - 0.1 * np.sin(0.75 * t) * np.exp(-0.1 * t)

# Parameters
t_start = 0
t_end = 5
num_steps = 50 # dt = 0.1   (change to test different values)
h = (t_end - t_start) / num_steps

# Initialize arrays to store results
t_values = np.linspace(t_start, t_end, num_steps + 1)
analytical_results = f(t_values)
approx_results = np.zeros_like(analytical_results)

# Initialize the initial value
approx_results[0] = f(t_start)

# Houbolt method (modified Euler method)
for i in range(1, num_steps + 1):
    k1 = h * f_prime(t_values[i - 1])
    k2 = h * f_prime(t_values[i - 1] + h)
    approx_results[i] = approx_results[i - 1] + 0.5 * (k1 + k2)

# Plot the results
plt.plot(t_values, analytical_results, label="Analytical Solution")
plt.plot(t_values, approx_results, label="Houbolt Method (Modified Euler)")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.title("Analytical Solution vs Houbolt Method (Modified Euler) Approximation")
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
def f(t):
    return np.exp(-0.1 * t) * np.sin(0.75 * t)

def f_prime(t):
    return 0.75 * np.exp(-0.1 * t) * np.cos(0.75 * t) - 0.1 * np.sin(0.75 * t) * np.exp(-0.1 * t)

def f_double_prime(t):
    return -0.15 * np.exp(-0.1 * t) * np.cos(0.75 * t) - 0.5625 * np.exp(-0.1 * t) * np.sin(0.75 * t) + 0.01 * np.exp(-0.1 * t) * np.sin(0.75 * t)

# Parameters
t_start = 0
t_end = 5
num_steps = 50 # dt = 0.1   (change to test different values)
h = (t_end - t_start) / num_steps
alpha = 0.25  # Newmark method parameter
beta = 0.5 * (1 + 2 * alpha)

# Initialize arrays to store results
t_values = np.linspace(t_start, t_end, num_steps + 1)
analytical_results = f(t_values)
approx_results = np.zeros_like(analytical_results)

# Initialize the initial values
approx_results[0] = f(t_start)
approx_velocity = f_prime(t_start)

# Newmark method (Average Acceleration)
for i in range(1, num_steps + 1):
    delta_t = t_values[i] - t_values[i - 1]
    
    a_n_minus_1 = f_double_prime(t_values[i - 1])
    a_n = f_double_prime(t_values[i])
    
    u_n_plus_1 = approx_results[i - 1] + delta_t * approx_velocity + (0.5 - beta) * delta_t ** 2 * a_n_minus_1
    v_n_plus_1 = approx_velocity + delta_t * (1 - alpha) * a_n_minus_1
    
    a_n_plus_1 = f_double_prime(t_values[i])
    
    u_n_plus_1 = u_n_plus_1 + beta * delta_t ** 2 * a_n_plus_1
    v_n_plus_1 = v_n_plus_1 + alpha * delta_t * a_n_plus_1
    
    approx_results[i] = u_n_plus_1
    approx_velocity = v_n_plus_1

# Plot the results
plt.plot(t_values, analytical_results, label="Analytical Solution")
plt.plot(t_values, approx_results, label="Newmark Method ")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.title("Analytical Solution vs Newmark Method Approximation")
plt.show()


# In[ ]:




