#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Some character recognition"""
import numpy as np
import matplotlib.pyplot as plt

print("Hello world!")
some_array = np.random.randint(0, 21, size=(20, 2))
print(some_array)

plt.scatter(some_array[:, 0], some_array[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot')

plt.show()
