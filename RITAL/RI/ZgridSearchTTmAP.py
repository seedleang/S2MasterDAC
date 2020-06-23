# -*- coding: utf-8 -*-

from Parser import Parser, Document
from Indexer import IndexerSimple
from Models import ModeleLangue, Okapi
from Query import QueryParser
from Metrics import Precision, Rappel, FMesure, AvgPrecision, ReciprocalRank, NDCG
from Eval import EvalIRModel
import numpy as np
import pandas as pd

#////////////////////////////////// Grid Search avec séparation train/test ///////

def trainTest(data,p):
    # Fonction de séparation des données.
    # Input : data: dictionnaire {id_doc: objet Query}, p: pourcentage de train
    n = len(data) # nombre total de documents
    iddocs = list(data.keys()) # identifiants     
    # Récupérer les identifiants qui serviront de données train
    idtrain = np.random.choice(iddocs,int(p*n),replace=False)
    # et créer les dictionnaires finaux
    train = {}
    test = {}
    for iddoc in iddocs:
        if iddoc in idtrain: train[iddoc] = data[iddoc]
        else: test[iddoc] = data[iddoc]
    return train, test

#////////////////////////////////// Mise en oeuvre. //////////////////////////////

file = "data/cisi/cisi"

# Création de la collection de documents et de requêtes.
parse = Parser(file+".txt")
coll = parse.getResult() 
indexcoll = IndexerSimple(coll) 
qp = QueryParser(file+".qry",file+".rel")
queries = qp.getQoll() 
# Séparation train-test :
train, test = trainTest(queries,0.8)
avgP = AvgPrecision() # mAP

# Jelinek-Mercer /////////////////////////////////////////////////////////////////

# Grille du paramètre lambda 
paramlambda = [0.1*i for i in range(11)] #[0, 0.1,..., 1]
res = []
# Pour chaque modèle testé, on calcule la moyenne d'average precision sur 
# l'ensemble des requêtes.
for l in paramlambda:
    e = EvalIRModel({"Jelinek": ModeleLangue(indexcoll,l)}, {"avgP": avgP}, train)
    res.append(e.evalAll()["avgP"]["Jelinek"][0]) 

# Estimation du meilleur paramètre.
best = paramlambda[np.argmax(res)]
print("Optimum: {}".format(best))

# Test en validation.
e = EvalIRModel({"Jelinek": ModeleLangue(indexcoll,best)}, {"avgP": avgP}, test)
print("Test en validation: {}".format(e.evalAll()["avgP"]["Jelinek"][0]))

# Okapi BM25 /////////////////////////////////////////////////////////////////////

# Définition des grilles de paramètres (k1 et b)
paramK1 = [0.1*i for i in range(8,19)] #[0.8, 0.9,..., 1.9]
paramb = [0.1*i for i in range(1,11)] #[0.1,..., 1]

res = [] # contiendra les valeurs d'avgP pour tous les modèles testés
liste = [] # contiendra les combinaisons de paramètres
for k1 in paramK1:
    for b in paramb:
        e = EvalIRModel({"Okapi": Okapi(indexcoll,k1,b)}, {"avgP": avgP}, train)
        res.append(e.evalAll()["avgP"]["Okapi"][0])
        liste.append((k1,b))

# Estimation du meilleur paramètre.
print(res)
best = liste[np.argmax(res)]
print("Optimum: {}".format(best))

# Test en validation. 
e = EvalIRModel({"Okapi": Okapi(indexcoll,best[0],best[1])}, {"avgP": avgP}, test)
print("Test en validation: {}".format(e.evalAll()["avgP"]["Okapi"][0]))