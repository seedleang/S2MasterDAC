# -*- coding: utf-8 -*-

from Parser import Parser, Document
from Indexer import IndexerSimple
from Models import ModeleLangue, Okapi
from Query import QueryParser
from Metrics import Precision, Rappel, FMesure, AvgPrecision, ReciprocalRank, NDCG
from Eval import EvalIRModel
import numpy as np
import pandas as pd

#////////////////////////////////// Grid Search avec séparation en k-folds ///////

def kFolds(data,k):
    # Fonction de séparation des données, à partir d'un dictionnaire 
    # {id_doc: objet Query} et du nombre k de folds voulus.
    taille = int(len(data)/k)
    iddocs = np.array(list(data.keys()))
    tmp = range(len(data))
    res = []
    for i in range(k-1):
        f = np.random.choice(tmp,taille,replace=False)
        res.append({})
        for iddoc in iddocs[f]:
            res[i][iddoc] = data[iddoc]
        tmp = list(set(tmp)-set(f))
    # Création du dernier fold avec les éléments surnuméraires. Sur 31 éléments et 
    # trois folds, le premier et le deuxième en contiendront 10, le dernier 11.
    res.append({})
    for iddoc in iddocs[tmp]:
        res[k-1][iddoc] = data[iddoc]
    return np.array(res)

def mergeFolds(folds):
    # Fusion des folds en un seul dictionnaire 
    # à partir d'une liste de dictionnaires à fusionner [{id_doc: objet Query}].
    if len(folds) <= 1: return folds
    res = {**folds[0], **folds[1]}
    for i in range(2,len(folds)):
        res.update(folds[i])
    return res
 
#////////////////////////////////// Mise en oeuvre. //////////////////////////////

file = "data/cisi/cisi"

parse = Parser(file+".txt")
coll = parse.getResult() 
indexcoll = IndexerSimple(coll) 
qp = QueryParser(file+".qry",file+".rel")
queries = qp.getQoll() 
# Séparation k-folds :
k = 5 # nombre de feuillets
folds = kFolds(queries,k)
avgP = AvgPrecision()

# Jelinek-Mercer /////////////////////////////////////////////////////////////////

# Grille du paramètre lambda 
paramlambda = [0.1*i for i in range(11)] #[0, 0.1,..., 1]
resBest = []
resTest = []
# Pour chaque modèle testé, on calcule la moyenne d'average precision sur 
# l'ensemble des requêtes, en faisant tourner les folds.
for h in range(k):
    # Fusion de tous les folds, moins celui qui sert de test. 
    train, test = mergeFolds(folds[list(set(range(k))-set([h]))]), folds[h]
    res = []
    for l in paramlambda:
        e = EvalIRModel({"Jelinek": ModeleLangue(indexcoll,l)}, {"avgP": avgP}, train)
        res.append(e.evalAll()["avgP"]["Jelinek"][0])
    # Estimation du meilleur paramètre pour ce modèle.
    resBest.append(paramlambda[np.argmax(res)])
    # Test en validation
    e = EvalIRModel({"Jelinek": ModeleLangue(indexcoll,0.1)}, {"avgP": avgP}, test)
    resTest.append(e.evalAll()["avgP"]["Jelinek"][0])

# Moyenne des meilleures valeurs du paramètre lambda
best = np.mean(resBest)
print("Optimum: {}".format(best)) 

# Résultat, 0.1 pour cisi, 0.9 pour cacm.

# Okapi BM25 //////////////////////////////////////////////////////////////////////

# Définition des grilles de paramètres (k1 et b)
paramK1 = [0.1*i for i in range(8,19)] #[0.8, 0.9,..., 1.9]
paramb = [0.1*i for i in range(1,11)] #[0.1,..., 1]

resk1 = []
resb = []
resTest = []
# Pour chaque modèle testé, on calcule la moyenne d'average precision sur 
# l'ensemble des requêtes, en faisant tourner les folds.
for h in range(k):
    train, test = mergeFolds(folds[list(set(range(k))-set([h]))]), folds[h]
    res = []
    liste = []
    for k1 in paramK1:
        for b in paramb:
            e = EvalIRModel({"Okapi": Okapi(indexcoll,k1,b)}, {"avgP": avgP}, train)
            res.append(e.evalAll()["avgP"]["Okapi"][0])
            liste.append((k1,b))
    # Estimation des meilleurs paramètres pour ce modèle
    best = liste[np.argmax(res)]
    resk1.append(best[0])
    resb.append(best[1])
    # Test en validation
    e = EvalIRModel({"Okapi": Okapi(indexcoll,best[0],best[1])}, {"avgP": avgP}, test)
    resTest.append(e.evalAll()["avgP"]["Okapi"][0])

# Moyenne des meilleures valeurs des paramètres k1 et b obtenus.
bestk1 = np.mean(resk1)
bestb = np.mean(resb)
print("Optimum: {}".format((bestk1,bestb)))

# Résultats finaux, 
#(1.72, 0.9) pour cacm
#(1.8, 0.88) pour cisi