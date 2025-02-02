from math import log
from math import fmod
from math import exp
from math import atan
from math import tan
import math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

#retourne la precision de x et y
def err(x,y):
    #convertir les nombres en chaine de caracteres pour determiner les chifres significatifs
    a,b=str(x),str(y)

    #i parcours les deux chaines et precision la precision calculer
    i,precision=0,0

    #on increment i tant qu'il y a des chiffres non significatifs
    while(i<len(a) and i<len(b) and a[i]==b[i] and (a[i]=="0" or a[i]==".")):
        i+=1

    #maintenant on increment la precision 
    while(i<len(a) and i<len(b) and a[i]==b[i]):
        if(a[i]!="."):
            precision+=1
        i+=1
    return precision

#calcule de ln(x) par l'algorithme cordic
def cordic_ln(x):
    init()
    k,y,p=0,0,1
    while (k<=6):
        t=1+10**(-k)
        while(x>=p*t):
            y+=ln[k] 
            p*=t
        k+=1
    eps=x/p-1
    y+=eps
    return y


#calcule de exp(x) par l'algorithme cordic
def cordic_exp(x):
    init()
    k,y,a=0,1,x
    while(k<=6):
        while(x>=ln[k]):
            x=x-ln[k]
            y=y+y*(10**(-k))
        k+=1
    return (y+y*x)


#calcule de arctan(x) par l'algorithme cordic
def cordic_arctan(x):
    init()
    k,y,r,j,l=0,1,0,0,0

    #Le développement limité de arctan(x)~x x->0 est une solution pour que k ne depasse pas 4
    if(x<0):
        x=-x
        j=1
    if(x>1):
        x=1/x
        l=1
    if(x<10**(-10)):
        return x
    while(k<=4):

        while(x<y*(10**(-k)) and  k<=4):
            k+=1
        t=10**(-k)
        xp=x-y*t
        y+=x*t
        x=xp
        r+=A[k]
    r+=x/y
    if(l==1):
        r=(pi/2)-r
    if(j==1):
        r=-r
    return r

# retourne x mod (pi)
def mod(x):
    while((x-pi)>=0):
        x-=pi
    return x

#calcule de tan(x) par l'algorithme cordic
def cordic_tan(x):
    init()
    j,z,b=0,0,0
    if(x<0):
        x=-x
        b=1
    if(x>pi):
        x= mod(x)
    if(x>(pi/2)):
        x=pi-x
        j=1
    if(x>(pi/4)):
        x=pi/2-x
        z=1
    k,n,d,a=0,0,1,x
    while(k<=4):
        while(x>=A[k]):
            t=10**(-k)
            x-=A[k]
            np=n+d*t
            d-=n*t
            n=np
        k+=1
    res=(n+x*d)/(d-x*n)
    if(j==1):
        res=-res
    if(z==1):
        res=1/res
    if(b==1):
        res=-res
    return res

#les deux tableaux L et A
def init():
    global ln
    ln=[log(2)]
    for i in range(1,7):
        ln.append(log(1+10**(-i)))
    global A
    A=[atan(1)]
    for i in range(1,6):
        A.append(atan(10**(-i)))



if __name__ == "__main__":
    # Tests pour ln et exp
    print("ln(2) ≈", cordic_ln(2))
    print("exp(1) ≈", cordic_exp(1))
    
    # Tests pour tan et arctan
    print("arctan(1) ≈", cordic_arctan(1))
    print("tan(0.5) ≈", cordic_tan(0.5))
