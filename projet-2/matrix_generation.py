import numpy as np

# def generate_symmetric_positive_definite_matrix(n):
#     A = np.random.rand(n, n)
#     return np.dot(A, A.T)

#calcule la valeur de la ligne i de A sans le diagonal et retourne -1 si on a pas de colonne correspond à i null sinon retourne la somme
def sum_lign(A,n,i):
    somme=0
    for j in range(n):
        if(i!=j):
            somme+=A[i][j]
    return somme


# Calcule la valeur de la colonne en optimisant les vérifications
def valeur_colonne(A, n, i):
    som = sum_lign(A, i)
    while True:
        x = np.random.uniform(0, A[i][i] - som)
        j = np.random.randint(0, n)
        if j != i:
            somme = som + x
            somme1 = sum_lign(A, j) + x
            if x > 0 and somme <= A[i][i] and somme1 <= A[j][j]:
                return x, j

#verifier si la ligne i de la matrice de taille n contient un 0 ou non
def ligne_null(A,n,i):
    
    #pour verifier les cases triangulaires superieurs
    for j in range(i+1,n):
        if(A[i][j]==0):
            return j
    return -1

#on utilise les matrices dominantes
def generate_symmetric_positive_definite_matrix(n):

    A=np.random.rand(n,n)
    A=np.dot(A,np.transpose(A))
    for i in range(n):
        somme_i=sum_lign(A,n,i)+1
        A[i][i]=np.random.uniform(somme_i+1,2*somme_i)
    return A

#fonction qui genere une fonction symetrique positive et definie creuses avec n la taille de la matrice et m le nombre des coefficients hors le diagonal qui sont non null 
def generate_sparse_symmetric_positive_definite_matrix(n, m_percentage):
    m = int((n * n - n) * m_percentage / 100)
    A = generate_symmetric_positive_definite_matrix(n)
    # on calcule le nombre des 0 dans A
    nbre_zero = np.count_nonzero(A == 0)
    
    if nbre_zero < m:
        # récuperer les indices des valeurs non nulles
        indices = np.argwhere(A != 0)
        # mélanger les indices
        np.random.shuffle(indices)
        k = 0
        while nbre_zero < m:
            # récupérer les indices de la valeur non nulle
            i, j = indices[k]
            if i != j:
                # mettre i,j et j,i à 0
                A[i][j] = 0
                A[j][i] = 0
                nbre_zero += 2
            k += 1
    elif nbre_zero > m:
        # récuperer les indices des valeurs nulles
        indices = np.argwhere(A == 0)
        # mélanger les indices
        np.random.shuffle(indices)
        k = 0
        while nbre_zero > m:
            # récupérer les indices de la valeur nulle
            i, j = indices[k]
            if i != j:
                # mettre i,j et j,i à une valeur aléatoire
                A[i][j] = np.random.uniform(0.1, A[i][i] + A[i][j] - sum_lign(A, n, i))
                A[j][i] = np.random.uniform(0.1, A[j][j] + A[j][i] - sum_lign(A, n, j))
                nbre_zero -= 2
            k += 1
    return A


def M_c(N):
    A=np.zeros((N**2,N**2))
    for i in range(N**2):
        A[i][i]=4
        if(i!=(N**2-1)):
            A[i][i+1]=-1
            A[i+1][i]=-1
        if(i<=(N*(N-1)-1)):
            A[i][i+N]=-1
        if(i>=N):
            A[i][i-N]=-1
    return A
