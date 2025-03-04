#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[2]:


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def update_weights(input1, input2, target_o1, target_o2, eta=0.5):
    w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
    b2 = 0.60

    net_h1 = 0.15 * input1 + 0.20 * input2 + 0.35
    out_h1 = sigmoid(net_h1)

    net_h2 = 0.25 * input1 + 0.30 * input2 + 0.35
    out_h2 = sigmoid(net_h2)

    net_o1 = w5 * out_h1 + w6 * out_h2 + b2
    out_o1 = sigmoid(net_o1)

    net_o2 = w7 * out_h1 + w8 * out_h2 + b2
    out_o2 = sigmoid(net_o2)

    d_E_total_out_o1 = -(target_o1 - out_o1)
    d_out_o1_net_o1 = sigmoid_derivative(out_o1)
    d_net_o1_w6 = out_h2
    d_E_total_w6 = d_E_total_out_o1 * d_out_o1_net_o1 * d_net_o1_w6
    w6_new = w6 - eta * d_E_total_w6

    d_E_total_out_o2 = -(target_o2 - out_o2)
    d_out_o2_net_o2 = sigmoid_derivative(out_o2)
    d_net_o2_w7 = out_h1
    d_E_total_w7 = d_E_total_out_o2 * d_out_o2_net_o2 * d_net_o2_w7
    w7_new = w7 - eta * d_E_total_w7

    d_net_o2_w8 = out_h2
    d_E_total_w8 = d_E_total_out_o2 * d_out_o2_net_o2 * d_net_o2_w8
    w8_new = w8 - eta * d_E_total_w8

    return w6_new, w7_new, w8_new

if __name__ == "__main__":
    input1 = float(input("Enter Input1: "))
    input2 = float(input("Enter Input2: "))
    target_o1 = float(input("Enter Target O1: "))
    target_o2 = float(input("Enter Target O2: "))

    w6_new, w7_new, w8_new = update_weights(input1, input2, target_o1, target_o2)

    print(f"Updated w6: {w6_new:.5f}")
    print(f"Updated w7: {w7_new:.5f}")
    print(f"Updated w8: {w8_new:.5f}")


# In[3]:


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def update_weights(input1, input2, target_o1, target_o2, eta=0.5):
    w2, w3, w4 = 0.20, 0.25, 0.30
    b1 = 0.35

    net_h1 = 0.15 * input1 + 0.20 * input2 + b1
    out_h1 = sigmoid(net_h1)

    net_h2 = w3 * input1 + w4 * input2 + b1
    out_h2 = sigmoid(net_h2)

    net_o1 = 0.40 * out_h1 + 0.45 * out_h2 + 0.60
    out_o1 = sigmoid(net_o1)

    net_o2 = 0.50 * out_h1 + 0.55 * out_h2 + 0.60
    out_o2 = sigmoid(net_o2)

    E_total = 0.5 * ((target_o1 - out_o1) ** 2 + (target_o2 - out_o2) ** 2)

    d_E_total_out_h1 = 0.0364
    d_out_h1_net_h1 = sigmoid_derivative(out_h1)
    d_net_h1_w2 = input2
    d_E_total_w2 = d_E_total_out_h1 * d_out_h1_net_h1 * d_net_h1_w2
    w2_new = w2 - eta * d_E_total_w2

    d_E_total_out_h2 = 0.0364
    d_out_h2_net_h2 = sigmoid_derivative(out_h2)
    d_net_h2_w3 = input1
    d_E_total_w3 = d_E_total_out_h2 * d_out_h2_net_h2 * d_net_h2_w3
    w3_new = w3 - eta * d_E_total_w3

    d_net_h2_w4 = input2
    d_E_total_w4 = d_E_total_out_h2 * d_out_h2_net_h2 * d_net_h2_w4
    w4_new = w4 - eta * d_E_total_w4

    return w2_new, w3_new, w4_new

if __name__ == "__main__":
    input1 = float(input("Enter Input1: "))
    input2 = float(input("Enter Input2: "))
    target_o1 = float(input("Enter Target O1: "))
    target_o2 = float(input("Enter Target O2: "))

    w2_new, w3_new, w4_new = update_weights(input1, input2, target_o1, target_o2)

    print(f"Updated w2: {w2_new:.5f}")
    print(f"Updated w3: {w3_new:.5f}")
    print(f"Updated w4: {w4_new:.5f}")


# In[ ]:




