# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod

class EvalMesure(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def evalQuery(self,liste,Query):
        # Evaluation d'une liste de documents classés et retournés par un modèle,
        # pour une requête donnée : comparaison avec les valeurs de référence pour la requête.
        pass

class Precision(EvalMesure):
    # Précision : capacité à ne retrouver QUE des documents pertinents, mesurée par tp/(tp+fp).
    def __init__(self,rang):
        super().__init__()
        # Nombre maximum de résultats considérés.
        self.rang = rang 
    def evalQuery(self,liste,req):
        tentative = set([source for source, score in liste[:self.rang]])
        reference = set(req.getPertinents())
        # "Vrais positifs" : comptage des réussites.
        truepos = len(tentative.intersection(reference)) 
        if len(tentative) == 0: return 0
        # Pondération par rapport au nombre total d'éléments jugés positifs (à tort ou à raison).
        return truepos/len(tentative)
    
class Rappel(EvalMesure):
    # Rappel : capacité à retrouver TOUS les documents pertinents, mesurée par tp/(tp+fn).
    def __init__(self,rang):
        super().__init__()
        # Nombre maximum de résultats considérés.
        self.rang = rang 
    def evalQuery(self,liste,req):
        tentative = set([source for source, score in liste[:self.rang]])
        reference = set(req.getPertinents())
        # "Vrais positifs" : comptage des réussites.
        truepos = len(tentative.intersection(reference)) 
        if len(reference) == 0: return 1
        # Pondération par rapport au nombre total d'éléments qu'il fallait juger tels.
        return truepos/len(reference) 
    
class FMesure(EvalMesure):
    # FMeasure : pour F1, moyenne harmonique de la précision et du rappel.
    def __init__(self,rang,beta=1):
        super().__init__()
        # Nombre maximum de résultats considérés.
        self.rang = rang 
        # Paramètre bêta à 1 par défaut (f1-score).
        self.beta = beta 
    def evalQuery(self,liste,req):
        tentative = set([source for source, score in liste[:self.rang]])
        reference = set(req.getPertinents())
        truepos = len(tentative.intersection(reference))
        # Sous-étape : précision et rappel
        P = truepos/len(tentative) if len(tentative)!=0 else 0 
        R = truepos/len(reference) if len(reference)!=0 else 0 
        # Application de la formule
        if (P+R) != 0: return (1+self.beta**2) * (P*R)/(self.beta**2*(P+R))
        return 0
    
class AvgPrecision(EvalMesure):
    # Moyenne de la précision,
    # avec une mesure binaire (0 ou 1) de la pertinence des documents sélectionnés.
    def __init__(self):
        super().__init__()
    def evalQuery(self,liste,req):
        reference = req.getPertinents()
        if len(reference) == 0: return 0
        # Attribution de la qualité de pertinence (oui/non) à la sélection,   
        pertinence = np.array([1 if source in reference else 0 for source,score in liste])
        # à sommer jusqu'au premier élément retourné à raison, 
        somme = 0
        for k in np.where(pertinence==1)[0]+1: # (ajustement des indices) 
            somme += sum(pertinence[:k])/k
        # et à ramener sur le nombre total d'éléments réellement pertinents.
        return somme/len(reference)

class ReciprocalRank(EvalMesure):
    # Moyenne du rang du premier document censément pertinent 
    # sur l’ensemble des requêtes.
    def __init__(self):
        super().__init__()
    def evalQuery(self,liste,req):
        reference = req.getPertinents()
        somme = 0
        for i in range(len(liste)):
            if liste[i][0] in reference:
                somme = i+1 # (ajustement des indices)
                break
        if somme==0: return 0
        else: return 1/somme 
    
class NDCG(EvalMesure):
    # nDCG : Normalised Discounted Cumulative Gain, prise en compte de l'ordre d'apparition
    # ndcg = dcg/idcg
    # avec dcg = sum(pertinence_i/log2(rang_i)) pour i document sélectionné ET pertinent
    #   et idcg = sum(pertinence_i/log2(rang_i)) dans le cas d'une sélection idéale
    # le tout avec une mesure binaire de la pertinence.
    def __init__(self,rang):
        super().__init__()
        # Nombre maximum de résultats considérés.
        self.rang = rang 
    def evalQuery(self,liste,req):
        if len(liste) == 0: return 0
        reference = req.getPertinents()
        # Limite 1: la taille totale de la sélection peut dépasser ou non le rang max déclaré.
        p = min(len(liste),self.rang)
        # Attribution de la qualité de pertinence (oui/non) à la sélection.
        pertinence = np.array([1 if source in reference else 0 for source, score in liste[:p]])
        # Limite 2: la taille totale de la référence peut dépasser ou non le rang max déclaré.
        p2 = min(len(reference),self.rang)
        # Mise en place du rapport.
        dcg = pertinence[0] + np.sum(pertinence[1:]/np.log2(range(2,p+1)))
        idcg = np.sum(np.ones(p2)/np.hstack((np.array([1]),np.log2(range(2,p2+1)))))
        if idcg == 0: return 0
        return dcg/idcg

# Autres métriques possibles : MRR (cours), etc.