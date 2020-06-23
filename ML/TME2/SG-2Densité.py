#!/usr/bin/env python
# coding: utf-8

#/////////////////////////////////////////////////////////////////////////////////////////////////// <imports nécessaires> ////

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.preprocessing import normalize
import random

#////////////////////////////////////////////////////////////////////////////////////////////////// </imports nécessaires> ////

#///////////////////////////////////////////////////////////////////////////////////////////// <visualisation des données> ////

# Setting géographique des données, la carte de Paris.
plt.ion()
lutece = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
# Elle a des coordonnées maximales et minimales.
xmin,xmax = 2.23,2.48  
ymin,ymax = 48.806,48.916 

# Affichage en contrôlant la transparence et l'échelle.
def show_map(alpha = None):
    plt.imshow(lutece,extent=[xmin,xmax,ymin,ymax],aspect=1.5,alpha = alpha)

# Liste des points d'intérêt et choix d'un type précis.
poidata = pickle.load(open("data/poi-paris.pkl","rb"))
allkeys = list(poidata.keys())
typepoi = allkeys[4]
print("Vous travaillez actuellement avec les "+typepoi+"s de Paris.\nVous vous intéresserez à leur répartition sur la ville.")
# On en récupère les coordonnées (latitude,longitude) pour tout afficher.
geoMat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geoMat[i,:]=v[0]

plt.figure(figsize=(12,7))
show_map()
plt.scatter(geoMat[:,1],geoMat[:,0],alpha=0.8,s=3,color='navajowhite')
plt.show()

# Le choix de travailler sur home_goods_stores est lié à l'existence de tout petits clusters bien localisés et très denses,
# il est utile de voir ce que les algorithmes en pensent.

#//////////////////////////////////////////////////////////////////////////////////////////// </visualisation des données> ////

#//////////////////////////////////////////////////////////////////////////////////////////// <estimation par histogramme> ////

# ··· La densité des POI peut être estimée en appliquant simplement une grille sur la carte :
# en choisissant la précision de cette grille - soit le nombre total de cases,
# on travaille alors par comptage des points pour donner une image globale de la fonction de densité.

class ModelGeoMat_Histo():
    def __init__(self, params_map, scalex, scaley):
        # Délimiter la zone de la carte.
        self.ymin,self.ymax,self.xmin,self.xmax = params_map
        # Précision de la grille de discrétisation.
        self.scalex = scalex
        self.scaley = scaley
        # Application de la discrétisation. 
        self.inter_x, self.stepx = np.linspace(self.xmin,self.xmax,self.scalex+1,retstep=True)
        self.inter_y, self.stepy = np.linspace(self.ymin,self.ymax,self.scaley+1,retstep=True)
        # Densité à calculer.
        self.densite = None
        
    def train(self,data):
        # Comptage, calcul des effectifs : 
        # parmi les données, combien tombent dans telle ou telle case ?
        effectifs = np.zeros((self.scalex,self.scaley))
        for dot in data:
            # Caractérisation de la position du point 
            # par rapport à la discrétisation de l'espace.
            posx = np.argmin(dot[0]>=self.inter_x)-1
            posy = np.argmin(dot[1]>=self.inter_y)-1
            effectifs[posx,posy] += 1
        # Déduction de la densité.
        v = self.stepx*self.stepy
        self.densite = effectifs/(v*effectifs.sum())
        
    def predict(self,prec=None):
        # L'estimation précise doit être permise :
        if prec:
            posx = np.argmin(prec[1]>=self.inter_x)
            posy = np.argmin(prec[0]>=self.inter_y)
            res = self.densite[posx-1,posy-1]
            return res
        # Sinon, donner une idée générale de la densité.
        return self.densite

#/////////////////////////////////////////////////////////////////////////////////////////// </estimation par histogramme> ////

#///////////////////////////////////////////////////////////////////////////////////////////////// <estimation par noyaux> ////

# ··· Cette fois, au lieu de travailler avec une grille indépendante des données, on décide d'étudier la densité avec une 
# notion de voisinage (fenêtre de Parzen). On évite ainsi le problème des frontières arbitraires : 
# quelle que soit sa position exacte par rapport à la grille, l'impact d'un point en tant que voisin d'un autre 
# ne dépendra que de la distance qui les sépare.
# ··· Pour savoir quoi retenir dans ce voisinage, et avec quel poids, on définit des types de "noyaux". 
# Deux méthodes à noyaux seront implémentées, les noyaux uniformes et les noyaux gaussiens.

class ModelGeoMat_Noyaux():
    def __init__(self,kernel,h):
        # Paramètre de lissage.
        self.h = h
        # Ici, plus besoin de discrétisation. La grille n'est appliquée que lors de l'affichage.
        # Les calculs se feront séparément sur chaque point des données dont on étudie le voisinage.
        self.thekernel = kernel
        self.densite = None
        self.data = None
        self.grid = None
        self.N = 0
        
    def noyauRosenblatt(self,x,y):
        # Couramment appelé noyau uniforme.
        return np.sum(np.where((np.abs(self.data[:,1]-x)<self.h/2)&(np.abs(self.data[:,0]-y)<self.h/2),1,0))
    
    def noyauGauss(self,x,y):
        # Où le poids dans le voisinage suit une répartition gaussienne.
        return np.sum(np.exp(-0.5*((((self.data[:,1]-x)/self.h)**2)+((self.data[:,0]-y)/self.h)**2))/(np.sqrt(2*np.pi)*self.h))
        
    def train(self,data):
        # La distinction train/test devient un peu obsolète ici.
        # De toute façon, le calcul n'est lisible qu'avec les repères de la grille.
        self.data = data
        self.N = len(data)
    
    def predict(self,grid):
        self.grid = grid
        self.densite = np.zeros((len(self.grid)))
        # Chaque point de la grille sert de centre. On étudie alors la façon dont les points des données se placent autour,
        # sans qu'il n'y ait de frontière à proprement parler : il est plutôt question de zones d'influence.
        for i, dot in enumerate(self.grid):
            if self.thekernel == 'uniforme':
                self.densite[i] = self.noyauRosenblatt(dot[0],dot[1])/(self.h**2)
            elif self.thekernel == 'gaussian':
                self.densite[i] = self.noyauGauss(dot[0],dot[1])
        return self.densite/self.N
    
#//////////////////////////////////////////////////////////////////////////////////////////////// </estimation par noyaux> ////

#////////////////////////////////////////////////////////////////////////////////////////////// <évaluations des densités> ////

# ··· Mise en oeuvre des trois types d'estimateurs.

xmin,xmax = 2.23,2.48  
ymin,ymax = 48.806,48.916 
params_map = xmin,xmax,ymin,ymax

plt.ion()
# Valeurs potentielles de la constante de lissage.
smooth = [0.042,0.019,0.005,0.001]
# Quantités potentielles de cases dans les grilles.
step = [5,10,15,20,42,100]
for h in smooth:
    for s in step:
        # Création du modèle.
        mhisto = ModelGeoMat_Histo(params_map,s,s)
        mnoyau1 = ModelGeoMat_Noyaux('uniforme',h)
        mnoyau2 = ModelGeoMat_Noyaux('gaussian',h)
        # Apprentissage.
        mhisto.train(geoMat)
        mnoyau1.train(geoMat)
        mnoyau2.train(geoMat)
        # Prédiction.
        aStep = plt.figure(figsize=(18,3))
        #///////////////////////////////////// OPTION 1 : les histogrammes /////////////////////////////////////////
        # Ici, la grille est intégrée à l'estimateur.
        res = mhisto.predict()    
        ax1 = aStep.add_subplot(131)
        ax1.title.set_text("Méthode des histogrammes, pas de %d" %(s))
        ax1 = show_map()
        ax1 = plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',alpha=0.3,aspect=1.5,origin ="lower",cmap='copper')
        ax1 = plt.colorbar()
        ax1 = plt.scatter(geoMat[:,1],geoMat[:,0],alpha=0.03,color='navajowhite')
        ax1 = plt.xlim([xmin,xmax])
        ax1 = plt.ylim([ymin,ymax])
        #///////////////////////////////////// OPTION 2 : fenêtre de Parzen ////////////////////////////////////////
        # Ne pas oublier de créer la grille au préalable.
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,s),np.linspace(ymin,ymax,s))
        grid = np.c_[xx.ravel(),yy.ravel()]
        res = mnoyau1.predict(grid).reshape(s,s)
        ax2 = aStep.add_subplot(132)
        ax2.title.set_text("Fenêtre de Parzen, pas de %d" %(s))
        ax2 = show_map()
        ax2 = plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',alpha=0.3,aspect=1.5,origin ="lower",cmap='pink')
        ax2 = plt.colorbar()
        ax2 = plt.scatter(geoMat[:,1],geoMat[:,0],alpha=0.03,color='pink')
        ax2 = plt.xlim([xmin, xmax])
        ax2 = plt.ylim([ymin, ymax])
        #///////////////////////////////////// OPTION 2bis : noyau gaussien ////////////////////////////////////////
        res = mnoyau2.predict(grid).reshape(s,s)
        ax3 = aStep.add_subplot(133)
        ax3.title.set_text("Noyau gaussien, pas de %d" %(s))
        ax3 = show_map()
        ax3 = plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',alpha=0.3,aspect=1.5,origin ="lower",cmap='pink')
        ax3 = plt.colorbar()
        ax3 = plt.scatter(geoMat[:,1],geoMat[:,0],alpha=0.03,color='pink')
        ax3 = plt.xlim([xmin, xmax])
        ax3 = plt.ylim([ymin, ymax])
        
#///////////////////////////////////////////////////////////////////////////////////////////// </évaluations des densités> ////

#////////////////////////////////////////////////////////////////////////////////////////// <génération pour vérification> ////

# ··· L'application de la méthode aux nightclubs est-elle réaliste ?

s=15
h=0.005

typepoi = allkeys[8]
print("Vous travaillez actuellement avec les "+typepoi+"s de Paris.\nVous tentez de vérifier votre estimation.")
# On en récupère les coordonnées (latitude,longitude) pour tout afficher.
geoMat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geoMat[i,:]=v[0]
    
mnoyau1 = ModelGeoMat_Noyaux('uniforme',h)
mnoyau1.train(geoMat)
xx,yy = np.meshgrid(np.linspace(xmin,xmax,s),np.linspace(ymin,ymax,s))
grid = np.c_[xx.ravel(),yy.ravel()]
res = mnoyau1.predict(grid).reshape(s,s)
normed_matrix = normalize(res.reshape(-1,1), axis=0, norm='l1')
tirage = np.array(random.choices(grid, weights=normed_matrix, k=400))

# Ajout d'une composante aléatoire (sans quoi la grille est un peu "visible")
std_x = (xmax-xmin)/60
std_y = (ymax-ymin)/60
random_x = np.array([random.gauss(0, std_x) for i in range(tirage.shape[0])])
random_y = np.array([random.gauss(0, std_y) for i in range(tirage.shape[0])])
tirage[:,0] += random_x
tirage[:,1] += random_y

# Affichage
real = plt.figure(figsize=(14,7))
ax1 = real.add_subplot(121)
show_map()
ax1.title.set_text("Fenêtre de Parzen, pas de %d, sigma de %.3f" %(s,h))
ax1 = plt.scatter(tirage[:,0],tirage[:,1],alpha=0.8,s=3,color='navajowhite')
ax2 = real.add_subplot(122)
show_map()
ax2.title.set_text("Situation initiale")
ax2 = plt.scatter(geoMat[:,1],geoMat[:,0],alpha=0.8,s=3,color='navajowhite')
plt.show()

# La réponse est oui.

#///////////////////////////////////////////////////////////////////////////////////////// </génération pour vérification> ////

#////////////////////////////////////////////////////////////////////////////////////////////////// <prédictions de notes> ////

# ··· On peut essayer de deviner la note d'un point en fonction de son emplacement.
# C'est une tâche de classification sur les notes manquantes des données.

# Estimateur de Nadaraya-Watson.
class Nadaraya():
    def __init__(self,kernel,h):
        self.h = h
        self.thekernel = kernel
        self.coord = None
        self.notes = None
        
    def fit(self,data):
        self.coord = data[:,:2]
        self.notes = data[:,2]
        
    def noyauRosenblatt(self,x,y):
        return np.where((np.abs(x-self.coord[:,0])<self.h/2)&(np.abs(y-self.coord[:,1])<self.h/2),1,0)
    
    def noyauGauss(self,x,y):
        return np.exp(-0.5*((((self.coord[:,0]-x)/self.h)**2)+((self.coord[:,1]-y)/self.h)**2))/(np.sqrt(2*np.pi)*self.h)

    def predict(self,test):
        predictions = np.zeros(len(test))
        for i, dot in enumerate(test):
            # On utilise le noyau gaussien qu'on connaît 
            # pour obtenir une pondération des notes visibles alentour.
            if self.thekernel == 'uniforme':
                pond = self.noyauRosenblatt(dot[0],dot[1])
            elif self.thekernel == 'gaussian':
                pond = self.noyauGauss(dot[0],dot[1])
            predictions[i] = np.sum(self.notes*pond)/np.sum(pond)
        return predictions
    
# Plus proches voisins.
class KNearestNeighbours():
    def __init__(self,k):
        self.k = k
        self.coord = None
        self.notes = None
        
    def fit(self,data):
        self.coord = data[:,:2]
        self.notes = data[:,2]
        
    def predict(self,test):
        predictions = np.zeros(len(test))
        for i, dot in enumerate(test):
            # On n'utilise aucune pondération particulière, mais le nombre de voisins 
            # considérés est limité par une valeur k en fonction de leur proximité.
            radius = np.linalg.norm(self.coord-dot[:2],axis=1)
            nearest = self.notes[np.argsort(radius)[:self.k]]
            predictions[i] = np.mean(nearest)
        return predictions

#///////////////////////////////////////////////////////////////////////////////////////////////// </prédictions de notes> ////

#//////////////////////////////////////////////////////////////////////////////////////// <mise en oeuvre des prédictions> ////

# Histoire d'avoir une matière un peu plus intéressante ET suffisamment de données :
typepoi = allkeys[11]
print("Vous travaillez actuellement avec les "+typepoi+"s de Paris. Vous vous intéressez à la note qu'on leur a donnée.")

# Pour séparer des arrays en fonction des tranches de valeurs.
def split(arr,cond):
    return [arr[cond],arr[~cond]]

# On récupère les données d'entraînement dans la matrice de départ.
starMat = np.zeros((len(poidata[typepoi]),3))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    starMat[i,0] = v[0][0]
    starMat[i,1] = v[0][1]
    starMat[i,2] = v[1] 
# Selon que la note est renseignée ou non :
starOK,starKO = split(starMat,(starMat[:,2]>=0))
# Pour obtenir un dégradé de couleurs sur la carte :
starOK1,starrest = split(starOK,starOK[:,2]<=1)
starOK2,starrest = split(starrest,starrest[:,2]<=2)
starOK3,starrest = split(starrest,starrest[:,2]<=3)
starOK4,starrest = split(starrest,starrest[:,2]<=4)
starOK5,starrest = split(starrest,starrest[:,2]<=5)

# Visualiser un peu la répartition des notes, qui sont généralement très bonnes,
# d'où l'ordre de superposition (les pires en dernier).
plt.figure(figsize=(15,10))
plt.title("Notes connues des bars de Paris.")
show_map(0.7)
plt.scatter(starOK5[:,1],starOK5[:,0],s=30,label="Entraînement",marker='*',color='floralwhite')
plt.scatter(starOK4[:,1],starOK4[:,0],s=30,label="Entraînement",marker='*',color='papayawhip')
plt.scatter(starOK3[:,1],starOK3[:,0],s=30,label="Entraînement",marker='*',color='navajowhite')
plt.scatter(starOK2[:,1],starOK2[:,0],s=30,label="Entraînement",marker='*',color='burlywood')
plt.scatter(starOK1[:,1],starOK1[:,0],s=30,label="Entraînement",marker='*',color='tan')
plt.scatter(starKO[:,1],starKO[:,0],alpha=0.8,s=40,label="Inconnues",marker='$?$',color='darkslategray')
plt.legend()
plt.show()

# Phase de préobservation terminée.
# Estimations et comparaison.
nadwat = Nadaraya('gaussian', 0.005)
nadwat.fit(starOK)
nadwat = nadwat.predict(starKO)
knn = KNearestNeighbours(8)
knn.fit(starOK)
knn = knn.predict(starKO)
print("Notes prédites par Nadaraya-Watson :   ",np.around(nadwat,decimals=2)[:14])
print("Notes prédites par NearestNeighbours : ", np.around(knn,decimals=2)[:14])

# Retracer le graphique, pour voir comment les notes prédites par les deux se fondent parmi les autres points.
# Remplir les notes manquantes avec Nadaraya-Watson :
count = 0
for i in range(len(starMat)):
    if starMat[i,2] < 0:
        starMat[i,2] = -(nadwat[count])
        count += 1
# Compter et séparer les valeurs prédites.
countsnad = np.zeros((5))
starKO5,starrest = split(starMat,starMat[:,2]<=-4)
countsnad[4] = len(starKO5)
starKO4,starrest = split(starrest,starrest[:,2]<=-3)
countsnad[3] = len(starKO4)
starKO3,starrest = split(starrest,starrest[:,2]<=-2)
countsnad[2] = len(starKO3)
starKO2,starrest = split(starrest,starrest[:,2]<=-1)
countsnad[1] = len(starKO2)
starKO1,starrest = split(starrest,starrest[:,2]<=0)
countsnad[0] = len(starKO1)

plt.figure(figsize=(16,10))
plt.title("Notes estimées des bars de Paris pour %s."%("Nadaraya-Watson"))
show_map(0.7)
plt.scatter(starOK5[:,1],starOK5[:,0],s=30,label="Entraînement",marker='*',color='floralwhite')
plt.scatter(starOK4[:,1],starOK4[:,0],s=30,label="Entraînement",marker='*',color='papayawhip')
plt.scatter(starOK3[:,1],starOK3[:,0],s=30,label="Entraînement",marker='*',color='navajowhite')
plt.scatter(starOK2[:,1],starOK2[:,0],s=30,label="Entraînement",marker='*',color='burlywood')
plt.scatter(starOK1[:,1],starOK1[:,0],s=30,label="Entraînement",marker='*',color='tan')
plt.scatter(starKO5[:,1],starKO5[:,0],s=100,label="Prédictions",marker='$!$',color='floralwhite')
plt.scatter(starKO4[:,1],starKO4[:,0],s=100,label="Prédictions",marker='$!$',color='papayawhip')
plt.scatter(starKO3[:,1],starKO3[:,0],s=100,label="Prédictions",marker='$!$',color='navajowhite')
plt.scatter(starKO2[:,1],starKO2[:,0],s=100,label="Prédictions",marker='$!$',color='burlywood')
plt.scatter(starKO1[:,1],starKO1[:,0],s=100,label="Prédictions",marker='$!$',color='tan')
plt.legend()
plt.show()

# Recommencer avec NearestNeighbours :
starMat = np.zeros((len(poidata[typepoi]),3))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    starMat[i,0] = v[0][0]
    starMat[i,1] = v[0][1]
    starMat[i,2] = v[1] 
# Remplir les notes manquantes avec NearestNeighbours :
count = 0
for i in range(len(starMat)):
    if starMat[i,2] < 0:
        starMat[i,2] = -(knn[count])
        count += 1
# Compter et séparer les valeurs prédites.
countsknn = np.zeros((5))
starKO5,starrest = split(starMat,starMat[:,2]<=-4)
countsknn[4] = len(starKO5)
starKO4,starrest = split(starrest,starrest[:,2]<=-3)
countsknn[3] = len(starKO4)
starKO3,starrest = split(starrest,starrest[:,2]<=-2)
countsknn[2] = len(starKO3)
starKO2,starrest = split(starrest,starrest[:,2]<=-1)
countsknn[1] = len(starKO2)
starKO1,starrest = split(starrest,starrest[:,2]<=0)
countsknn[0] = len(starKO1)

plt.figure(figsize=(16,10))
plt.title("Notes estimées des bars de Paris pour %s."%("NearestNeighbours"))
show_map(0.7)
plt.scatter(starOK5[:,1],starOK5[:,0],s=30,label="Entraînement",marker='*',color='floralwhite')
plt.scatter(starOK4[:,1],starOK4[:,0],s=30,label="Entraînement",marker='*',color='papayawhip')
plt.scatter(starOK3[:,1],starOK3[:,0],s=30,label="Entraînement",marker='*',color='navajowhite')
plt.scatter(starOK2[:,1],starOK2[:,0],s=30,label="Entraînement",marker='*',color='burlywood')
plt.scatter(starOK1[:,1],starOK1[:,0],s=30,label="Entraînement",marker='*',color='tan')
plt.scatter(starKO5[:,1],starKO5[:,0],s=100,label="Prédictions",marker='$!$',color='floralwhite')
plt.scatter(starKO4[:,1],starKO4[:,0],s=100,label="Prédictions",marker='$!$',color='papayawhip')
plt.scatter(starKO3[:,1],starKO3[:,0],s=100,label="Prédictions",marker='$!$',color='navajowhite')
plt.scatter(starKO2[:,1],starKO2[:,0],s=100,label="Prédictions",marker='$!$',color='burlywood')
plt.scatter(starKO1[:,1],starKO1[:,0],s=100,label="Prédictions",marker='$!$',color='tan')
plt.legend()
plt.show()

# Entourer en post-traitement les régions qui diffèrent, comme par exemple le regroupement qui change de couleur
# à l'extrême-nord / centre. Plus de données sur ce qui a été fait par les deux algorithmes :

print("Répartition des notes selon Nadaraya-Watson :      0··1 | %d,   1··2 | %d,   2··3 | %d,   3··4 | %d,   4··5 | %d"% (countsnad[0],countsnad[1],countsnad[2],countsnad[3],countsnad[4]))
print("Répartition des notes selon NearestNeighbours :    0··1 | %d,   1··2 | %d,   2··3 | %d,   3··4 | %d,   4··5 | %d"% (countsknn[0],countsknn[1],countsknn[2],countsknn[3],countsknn[4]))

#/////////////////////////////////////////////////////////////////////////////////////// </mise en oeuvre des prédictions> ////