#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt

# Original function
def fun(x):
    return 1.55 * x**4 - 12.2 * x**3 + 30.4 * x**2 - 25.8 * x + 2.3

# x values
x = np.arange(-2.5, 2.6, 0.1)

# penalty parameter
k = 100


f = fun(x)

# Plot f(x)
plt.plot(x, f, label='Original Function')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axvline(1, color='black')

# x in different ranges
i_neg = x[x <= 0]
i_pos = x[x >= 1]
i_x = x[(0 < x) & (x < 1)]

f_neg = fun(i_neg) + k * (i_neg)**2
plt.plot(i_neg, f_neg, label='x<0')

f_x = fun(i_x)
plt.plot(i_x, f_x, label=' x:[0,1]')

f_pos = fun(i_pos) + k * (i_pos - 1)**2
plt.plot(i_pos, f_pos, label='x>0')

# Add legend
plt.legend()
plt.text(0.8, 40, f'k = {k}', fontsize=12, color='red')
plt.xlabel('x^S')
plt.ylabel('function value')
plt.ylim(-20, 100)
plt.xlim(-2.5, 2.5)

plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt

# Original function
def fun(x):
    return 1.55 * x**4 - 12.2 * x**3 + 30.4 * x**2 - 25.8 * x + 2.3

# x values
x = np.arange(-2.5, 2.6, 0.1)

# penalty parameter
k = 1000

f = fun(x)

# Plot f(x)
plt.plot(x, f, label='Original Function')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axvline(1, color='black')

# x in different ranges
i_neg = x[x <= 0]
i_pos = x[x >= 1]
i_x = x[(0 < x) & (x < 1)]

f_neg = fun(i_neg) + k * i_neg**2
plt.plot(i_neg, f_neg, label='x<0')

f_x = fun(i_x)
plt.plot(i_x, f_x, label=' x:[0,1]')

f_pos = fun(i_pos) + k * (i_pos - 1)**2
plt.plot(i_pos, f_pos, label='x>0')

# Add legend
plt.legend()
plt.text(0.8, 40, f'k = {k}', fontsize=12, color='red')
plt.xlabel('x^S')
plt.ylabel('function value')
plt.ylim(-20, 100)
plt.xlim(-2.5, 2.5)

plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Original function
def fun(x):
    return 1.55 * x**4 - 12.2 * x**3 + 30.4 * x**2 - 25.8 * x + 2.3

# x values
x = np.arange(-2.5, 2.6, 0.1)

# penalty parameter
k = 10000


f = fun(x)

# Plot f(x)
plt.plot(x, f, label='Original Function')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axvline(1, color='black')

# x in different ranges
i_neg = x[x <= 0]
i_pos = x[x >= 1]
i_x = x[(0 < x) & (x < 1)]

f_neg = fun(i_neg) + k * i_neg**2
plt.plot(i_neg, f_neg, label='x<0')

f_x = fun(i_x)
plt.plot(i_x, f_x, label=' x:[0,1]')

f_pos = fun(i_pos) + k * (i_pos - 1)**2
plt.plot(i_pos, f_pos, label='x>0')

# Add legend
plt.legend()
plt.text(0.8, 40, f'k = {k}', fontsize=12, color='red')
plt.xlabel('x^S')
plt.ylabel('function value')
plt.ylim(-200, 200)
plt.xlim(-2.5, 2.5)

plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Original function
def fun(x):
    return 1.55 * x**4 - 12.2 * x**3 + 30.4 * x**2 - 25.8 * x + 2.3

# x values
x = np.arange(-2.5, 2.6, 0.1)

# penalty parameter
k = 100000


f = fun(x)

# Plot f(x)
plt.plot(x, f, label='Original Function')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axvline(1, color='black')

# x in different ranges
i_neg = x[x <= 0]
i_pos = x[x >= 1]
i_x = x[(0 < x) & (x < 1)]

f_neg = fun(i_neg) + k * i_neg**2
plt.plot(i_neg, f_neg, label='x<0')

f_x = fun(i_x)
plt.plot(i_x, f_x, label=' x:[0,1]')

f_pos = fun(i_pos) + k * (i_pos - 1)**2
plt.plot(i_pos, f_pos, label='x>0')

# Add legend
plt.legend()
plt.text(0.8, 40, f'k = {k}', fontsize=12, color='red')
plt.xlabel('x^S')
plt.ylabel('function value')
plt.ylim(-20, 40)
plt.xlim(-2.5, 2.5)

plt.show()


# In[ ]:




