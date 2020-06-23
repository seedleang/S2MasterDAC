# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import pandas

class EvalIRModel(object):
    def __init__(self,dictModels,dictMetrics,queries):
        self.dictModels = dictModels # dictionnaire {nom_modele: objet IRModel}
        self.dictMetrics = dictMetrics # dictionnaire {nom_mesure: objet EvalMesure}
        self.queries = queries # dictionnaire {id_query: objet Query}
        
    def evalSimple(self,idq,mod,met):
        # Score d'évaluation de la query pour la combinaison (modèle mod, métrique met).
        liste = self.dictModels[mod].getRanking(self.queries[idq].getTexte())
        return self.dictMetrics[met].evalQuery(liste,self.queries[idq])
    
    def evalSingleQuery(self,idq):
        # Dictionnaire de dictionnaires contenant le score d'évaluation d'une query pour chaque
        # combinaison (modèle, métrique) possible.
        evaluate = {}
        for met in self.dictMetrics.keys():
            evaluate[met] = {mod:self.evalSimple(idq,mod,met) for mod in self.dictModels.keys()}
        return evaluate
    
    def evalSingleComb(self,mod,met):
        # Tuple contenant une évaluation des performances (moyenne et écart-type) 
        # pour la combinaison modèle/métrique passée en entrée.
        evaluate = np.array([self.evalSimple(idq,mod,met) for idq in self.queries.keys()])
        # Parcourir toutes les requêtes et évaluer les résultats, pour finalement caractériser cette combinaison. 
        return (np.mean(evaluate),np.std(evaluate))
    
    def evalAll(self):
        # Dictionnaire de dictionnaires contenant une évaluation des performances (moyenne + écart-type) 
        # de chaque combinaison modèle/métrique possible.
        evaluate = {}
        for met in self.dictMetrics.keys():
            evaluate[met] = {}
            for mod in self.dictModels.keys():
		# Parcourir toutes les requêtes et reporter les résultats. 
                underev = np.array([self.evalSimple(idq,mod,met) for idq in self.queries.keys()])
                evaluate[met][mod] = (np.round(np.mean(underev)*100,3),np.round(np.std(underev)*100,3))
        return evaluate
    
    def tTest(self,X,Y,alpha):
        # Déterminer si deux modèles sont similaires. X et Y (array) contiennent 
        # les scores pour les requêtes évaluées des deux côtés. alpha est un facteur de "risque".
        # La fonction renvoie une t-test value, une valeur critique et une réponse finale :
        # True (modèles similaires), False (modèles trop différents).
        n = len(X)
        meanX = np.mean(X) 
        meanY = np.mean(Y)
        sX = np.sum((X-meanX)**2)/(n-1)
        sY = np.sum((Y-meanY)**2)/(n-1)
        z = (meanX-meanY)/(np.sqrt((sX+sY)/n)) # t-test value
        df = 2*n - 2 # degré de liberté
        cv = scipy.stats.t.ppf(1.0 - alpha/2, df) # critical value
        return z, cv, np.abs(z) <= cv
    
    def finalEv(self): 
        # L'affichage terminal.
        whole = self.evalAll()
        whole = pandas.DataFrame.from_dict(whole)
        print(whole)
        return whole