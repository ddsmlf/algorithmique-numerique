import math

def log_2_approximation(p):
    log_2_reel=math.log(2) # Calcul de la valeur réelle du logarithme népérien de 2
    log_2_machine = 0.0
    n = 1 # Compteur d'itérations
    precision = 10 ** (-p-2) # la précision à partir duquelle les termes de la série n'affectent plus le résultat voulue( approximation à p décimales) 
    term = 1 # Premier terme de la série 

    # une boucle qui continue d’ajouter les termes de la série tant que leurs valeurs absolues est supérieur à la précision définie précédemment 
    while abs(term) > precision:
        term = ((-1)**(n + 1)) / n
        log_2_machine += term  # Ajout du terme à l'approximation courante
        n += 1

    # arrondir l'approximation  sur p décimales
    factor = 10**p # Facteur d'échelle pour l'arrondi
    shifted_value = log_2_machine * factor # Décalage pour préparer l'arrondi
    if shifted_value - math.trunc(shifted_value) >= 0.5: # Test d'arrondi
        rounded_result = (math.trunc(shifted_value) + 1) / factor # Arrondi supérieur
    else:
        rounded_result = math.trunc(shifted_value) / factor # Arrondi inférieur
        
    # Calcul de l'erreur relative entre l'approximation arrondi et la vraie valeur de log(2)
    error = abs((log_2_reel - rounded_result) / log_2_reel)
    
    return rounded_result, error
    
    
if __name__ == "__main__":
    p = 5
    approximation, error = log_2_approximation(p)
    print(f"Log(2) approximé à {p} décimales: {approximation}")
    print(f"Erreur relative: {error}")
