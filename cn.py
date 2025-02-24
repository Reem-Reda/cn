#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random

def tanh_activation(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1
    
def forward_pass(x1, x2, w1, w2, w3, w4, w5, w6, b1, b2):
    h1_input = w1 * x1 + w2 * x2 + b1
    h1_output = tanh_activation(h1_input)
    
    h2_input = w3 * x1 + w4 * x2 + b1
    h2_output = tanh_activation(h2_input)
    
    output_input = w5 * h1_output + w6 * h2_output + b2
    output = tanh_activation(output_input)
    
    return h1_output, h2_output

w1, w2, w3, w4, w5, w6 = [random.uniform(-0.5, 0.5) for _ in range(6)]

b1, b2 = 0.5, 0.7

x1, x2 = 0.6, -0.1

h1_output, h2_output = forward_pass(x1, x2, w1, w2, w3, w4, w5, w6, b1, b2)

print("Output of h1:", h1_output)
print("Output of h2:", h2_output)



# In[ ]:




