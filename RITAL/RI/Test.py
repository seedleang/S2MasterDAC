# -*- coding: utf-8 -*-

from Parser import Parser, Document
from Indexer import IndexerSimple
from Weighters import Weighter1, Weighter2, Weighter3, Weighter4, Weighter5
from Models import Vectoriel, ModeleLangue, Okapi
from Query import QueryParser
from Metrics import Precision, Rappel, FMesure, AvgPrecision, ReciprocalRank, NDCG
from Eval import EvalIRModel
import numpy as np
import pandas as pd


# Phase de test. 
file = "data/cisi/cisi"
# file = "data/cacm/cacm"

print("Phase de test sur la collection CISI.\n\n                        ***")

# --------------------------------------------------------------------------- parsing + indexation ---------------------------

print("\n                     PARSER ///////////////////////////////////////////////////////////////////////////////////////")
print("\nParser la collection et afficher les trois premiers documents, puis le tout dernier.")
parse = Parser(file+".txt")
coll = parse.getResult()
print("\nAffichage de la collection. \n")
for document in list(coll.keys())[:3]:
     coll[document].snippet()
fin = 2460 # fin = 4204 pour cacm
coll[fin].snippet()
print("Terminé.\n                        ***")

print("\n                   INDEXER ////////////////////////////////////////////////////////////////////////////////////////")
print("\nTests de l'indexeur et des fonctions associées.")
indexcoll = IndexerSimple(coll)
print("\nTaille de la collection : ", indexcoll.getSize()[0])
print("\nScoring du dernier document : ", indexcoll.getTfIDFsForDoc(fin))
print("\nListe partielle du vocabulaire : ", list((indexcoll.getInverted()).keys())[8000:9000])
print("\nTerminé.\n                        ***")
     
# --------------------------------------------------------------------------- modèles ----------------------------------------

print("\n                     MODELS ///////////////////////////////////////////////////////////////////////////////////////")
print("\nTests rapides des modèles et appréciation des résultats à l'oeil nu.")
w = Weighter5(indexcoll) # La pondération la plus compliquée.
v = Vectoriel(indexcoll,w,True) # Score cosinus.
# Modèle Jelinek-Mercer *AVEC OPTIMISATION CISI*, cf. bonus en annexe.
m = ModeleLangue(indexcoll,0.1)
# Valeur cacm : 0.9
# Okapi BM25 *AVEC OPTIMISATION CISI*, cf. bonus en annexe.
o = Okapi(indexcoll,1.8,0.88) 
# Valeurs cacm : 
print("\nModèles disponibles : Vectoriel, Jelinek, Okapi.")
mods = {"Vectoriel":v, "Jelinek":m, "Okapi":o}

req = "The present study is a history of the DEWEY Decimal Classification.abroad" # version cisi
# req = "Computer Language : dramatic speed shallow or deep" # version cacm
print("\nRequête : \"{}\"".format(req))

print("\nSolutions pour le modèle vectoriel :\n")
#print(list(v.getScores(req).items())[0:20],"\n")
print(v.getRanking(req)[0:20])
print("\nSolutions pour le modèle de langue :\n")
#print(list(m.getScores(req).items())[0:20],"\n")
print(m.getRanking(req)[0:20])
print("\nSolutions pour le modèle Okapi :\n")
#print(list(o.getScores(req).items())[0:20],"\n")
print(o.getRanking(req)[0:20])

print("\nVérification rapide des documents récurrents :\n")
# Vérifications cisi
print(indexcoll.getTfsForDoc(1))
print(indexcoll.getTfsForDoc(260))
print(indexcoll.getTfsForDoc(354))
print(indexcoll.getTfsForDoc(282))
print(indexcoll.getTfsForDoc(1152))
print(indexcoll.getTfsForDoc(2287))
# Vérifications cacm
# print(indexcoll.getStrDoc(3801))
# print(indexcoll.getTfsForDoc(4101))
# print(indexcoll.getTfsForDoc(3803))
# print(indexcoll.getStrForDoc(198))

print("\nTerminé.\n                        ***")

# --------------------------------------------------------------------------- requêtes ---------------------------------------

print("\n                    QUERIES ///////////////////////////////////////////////////////////////////////////////////////")
print("\nParsing et préparation des requêtes.")
qp = QueryParser(file+".qry",file+".rel")
queries = qp.getQoll()
print("\nDémonstration rapide de cinq requêtes :\n")
for q in list(queries.values())[37:42]:
    q.showQuery()
print("\nTerminé.\n                        ***")

# --------------------------------------------------------------------------- mesures ----------------------------------------

print("\n                    METRICS ///////////////////////////////////////////////////////////////////////////////////////")
print("\nMise en place des métriques possibles.")
p = 50 # Rang maximum des documents à prendre en compte.
prec = Precision(p)
rapp = Rappel(p)
fmes = FMesure(p)
avgP = AvgPrecision()
rank = ReciprocalRank()
ndcg = NDCG(p)

mets = {"Precision":prec, "Recall":rapp, "FScore":fmes, "AvgPrecision": avgP, "RecipRank": rank, "NDCG":ndcg}
print("\nMétriques disponibles : Precision, Recall, FScore, AvgPrecision, RecipRank, NDCG.")
print("\nTerminé.\n                        ***")

# --------------------------------------------------------------------------- évaluation -------------------------------------

print("\n                 ASSESSMENT ///////////////////////////////////////////////////////////////////////////////////////")
print("\nComparaison des outils de recherche via plusieurs métriques.")
e = EvalIRModel(mods,mets,queries)
e.finalEv()
print("\nTerminé.\n                        ***")

df = pd.DataFrame(e.evalAll())

# --------------------------------------------------------------------------- à suivre ---------------------------------------

# T-tests
print("\nT-test isolé.")
modeleX = 'Okapi'  
modeleY = 'Vectoriel'
mesure = 'RecipRank'

X = np.array([e.evalSimple(idq,modeleX,mesure) for idq in queries.keys()])
Y = np.array([e.evalSimple(idq,modeleY,mesure) for idq in queries.keys()])

z, cv, res = e.tTest(X,Y,0.05)
print("t-test value: {}".format(z))
print("critical value: {}".format(cv))
print("Selon {}, {} et {} sont-ils similaires ? {}".format(mesure, modeleX, modeleY, res))

print("\nChanger la collection et adapter les valeurs liées pour réessayer. Elles sont disponibles en commentaire.\nTerminé.                       ***")