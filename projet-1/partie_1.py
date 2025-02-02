import math as m
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

def rp(x, p):
    """Représentation décimale réduite d'un nombre x avec p chiffres significatifs."""
    if x == 0:
        return 0.0 
    getcontext().prec = p
    x = Decimal(x)
    return float(Decimal(format(x, f".{p}g")))

def simule_add(n1, n2, precision):
    """Simule une addition en représentation décimale réduite."""
    return rp(rp(n1, precision) + rp(n2, precision), precision)

def simule_mul(n1, n2, precision):
    """Simule une multiplication en représentation décimale réduite."""
    return rp(rp(n1, precision) * rp(n2, precision), precision)

def err_add(x, y, precision):
    """Erreur relative pour l'addition."""
    return abs((x + y) - simule_add(x, y, precision)) / abs(x + y)

def err_mul(x, y, precision):
    """Erreur relative pour la multiplication."""
    return abs((x * y) - simule_mul(x, y, precision)) / abs(x * y)

def plot(x, y, precision):
    """Trace les erreurs relatives pour l'addition et la multiplication."""
    fig, axs = plt.subplots(2, figsize=(10, 8))
    end = 25

    x_values = [10**(-i) for i in range(1, end)]  # Correction de l'échelle logarithmique
    y_values_add = [err_add(x, y * val, precision) for val in x_values]
    y_values_mul = [err_mul(x, y * val, precision) for val in x_values]

    # Erreur Addition
    axs[0].plot(x_values, y_values_add, marker='o', linestyle='-', color='b', label='Erreur d\'addition')
    axs[0].set_title('Erreur relative dans l\'addition', fontsize=14)
    axs[0].set_xlabel('Ordre de grandeur de y (échelle logarithmique)', fontsize=12)
    axs[0].set_ylabel('Erreur relative', fontsize=12)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", ls="--")
    axs[0].legend()

    # Erreur Multiplication
    axs[1].plot(x_values, y_values_mul, marker='o', linestyle='-', color='r', label='Erreur de multiplication')
    axs[1].set_title('Erreur relative dans la multiplication', fontsize=14)
    axs[1].set_xlabel('Ordre de grandeur de y (échelle logarithmique)', fontsize=12)
    axs[1].set_ylabel('Erreur relative', fontsize=12)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", ls="--")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    number = 0.0001
    print(f"number = {number}")
    precision = 5
    result = rp(number, precision)
    print(f"Représentation décimale réduite : {result}")
    
    n1 = 0.0005684868468
    n2 = 785.0999
    print(f"Somme de {n1} + {n2} = {simule_add(n1, n2, precision)} avec une erreur de {err_add(n1, n2, precision)}")
    print(f"Multiplication de {n1} * {n2} = {simule_mul(n1, n2, precision)} avec une erreur de {err_mul(n1, n2, precision)}")

    
    n1 = 1.0
    n2 = 4678.0
    plot(n1, n2, precision)
