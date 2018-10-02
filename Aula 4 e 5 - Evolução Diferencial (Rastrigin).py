#Equipe: Nayara Thaiza, Mariana Leandra, Samuel Soares

from scipy.optimize import differential_evolution
import numpy as np
import math
import matplotlib.pyplot as plt

def rastrigin(X):
    A = 10
    return A*4 + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])

limites = [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]
resultado = differential_evolution(rastrigin,limites)
print(resultado.x)
print(resultado.fun)