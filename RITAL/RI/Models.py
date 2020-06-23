# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import TextRepresenter as tr
import numpy as np

class IRModel(ABC):
    def __init__(self,refindex):
        # Un modèle à préciser de recherche d'information.
        self.ref = refindex  # classe IndexerSimple
        self.inds = refindex.getIndex()
        self.inv = refindex.getInverted()
        self.invn = refindex.getInvertedN()
    
    @abstractmethod
    def getScores(self,query):
        # Scores des documents pour une requête donnée.
        pass
    
    def getRanking(self,query):
        # Liste (iddoc/score) dans l'ordre décroissant pour les appariements valides (=/= 0).
        docs_scores = self.getScores(query)
        return [(k,v) for k,v in sorted(docs_scores.items(), key=lambda item: item[1], reverse=True) if v!=0]

class Vectoriel(IRModel):
    def __init__(self,refindex,refweighter,normalized=False):
        # Modèle fondamental.
        super().__init__(refindex)
        self.weightype = refweighter # classe Weighter
        self.normalized = normalized # si True, score cosinus, sinon produit scalaire
        if self.normalized:
            # Optimisation : calcul des normes d'emblée pour tous les documents.
            self.normDs = {source:np.sqrt(np.sum(np.power(list(self.weightype.getWeightsForDoc(source).values()),2))) for source in self.inds.keys()}
    def getScores(self, query):
        # Pondération des mots de la requête.
        qweights = self.weightype.getWeightsForQuery(query)
        qstems = qweights.keys()
        # Préparation de la valeur à retourner, init 0.
        scores = dict.fromkeys(range(1,self.ref.getSize()[0]+1),0)
        if not self.normalized:
            # Produit scalaire.
            for stem in qstems:
                sweights = self.weightype.getWeightsForStem(stem)
                for source, o in sweights.items():
                    scores[source] += o*qweights[stem]
        else:
            # Score cosinus.
            normQ = np.sqrt(np.sum(np.power(list(qweights.values()),2)))
            for stem in qstems:
                sweights = self.weightype.getWeightsForStem(stem)
                for source, o in sweights.items():
                    scores[source] += (o*qweights[stem])/(normQ+self.normDs[source])
        return scores

class ModeleLangue(IRModel):
    def __init__(self,refindex,l=0.8):
        # Ici, on obtient le score d'un document 
        # en pondérant la fréquence relative du mot dans le document 
        # et la fréquence relative du mot dans la collection.
        # La constante de pondération s'appelle lambda.
        # produit [ P(t|d) = (1−l)*P(t|Md) + l*tP(t|MC) pour t dans q ]
        # avec      P(t|Md) = tf(t,d)/nb_mots_dans_le_document
        #           P(t|MC) = freq_de_t_dans_la _collection/nb_de_mots_dans_la_collection
        super().__init__(refindex)
        self.l = l
        
    def getScores(self,query):
        # Prétraitement de la requête.
        p = tr.PorterStemmer()
        qstems = p.getTextRepresentation(query).keys()          
        scores = {}
        # Pour chaque élément de la requête, ayant telle fréquence d'apparition dans la collection,
        # (qu'on appelle le facteur de lissage)
        for stem in qstems:
            pcoll = sum(self.inv[stem].values())/self.ref.getSize()[2] if stem in self.inv.keys() else 0
            # on renseigne sa probabilité d'être dans chaque document (via l'index normalisé),
            for source in self.inds.keys():
                pdoc = self.invn[stem][source] if stem in self.inds[source].keys() else 0
                # avant d'opérer la multiplication (log-somme pour faire joli)
                if ((1-self.l)*pdoc+self.l*pcoll)>0:
                    scores[source] = scores.get(source,0)+np.log((1-self.l)*pdoc+self.l*pcoll)
        return scores

class Okapi(IRModel):
    def __init__(self,refindex,k1=1.2,b=0.75):
        # Pour BM25, le score d'un document est donné en combinant 
        # les idf et les tf des mots de la requête dans la collection. 
        # L'un des éléments nouveaux est la prise en compte de la taille que prend chaque
        # document dans la collection par rapport à la longueur moyenne des autres.
        super().__init__(refindex)
        self.k1 = k1
        self.b = b
        # Longueur moyenne des documents.
        self.avgdl = np.mean([sum(self.inds[source].values()) for source in self.inds.keys()])
        
    def getScores(self,query):
        # Prétraitement de la requête.
        p = tr.PorterStemmer()
        qstems = p.getTextRepresentation(query).keys()
        # Valeur de retour, init 0.
        scores = dict.fromkeys(range(1,self.ref.getSize()[0]+1),0)
        # Etablir d'emblée les longueurs de tous les documents.
        lendocs = {source:sum(self.inds[source].values()) for source in self.inds.keys()}
        for stem in qstems:
            # Calculer les idf pour la requête,
            idf = self.ref.computeIdf(stem)
            for source in self.inds.keys():
                if stem in self.inds[source].keys():
                    # puis les tf si applicables
                    tf = self.inds[source][stem]
                    # et augmenter la pertinence du document lié.
                    scores[source] += idf*(tf/(tf+self.k1*(1-self.b+self.b*(lendocs[source]/self.avgdl))))
        return scores