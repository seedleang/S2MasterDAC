# -*- coding: utf-8 -*-

import re
import numpy as np

class Query(object):
    def __init__(self,idquery):
        self.idquery = idquery
        self.texte = ""
        # Liste des documents jugés pertinents.
        self.pertinents = [] 
        
    def showQuery(self):
        print(self.idquery,"\n",self.texte)
    
    # Getters et setters utiles.
    def getIdQuery(self):
        return self.idquery
    def getTexte(self):
        return self.texte
    def getPertinents(self):
        return self.pertinents
    def addTexte(self,texte):
        self.texte += texte
    def addPertinents(self,pertinent):
        self.pertinents += [pertinent]
        
class QueryParser(object):
    def __init__(self,queries,pertinents):
        self.fquery = queries # Nom du fichier contenant les requêtes.
        self.fperti = pertinents # Nom du fichier contenant les id pertinents.
        self.qoll = self.parsing()  
        
    def parsing(self):
        # Remplissage des champs de la collection de requêtes 
        # avec des objets Query.
        qoll = {}
        # Lecture du texte des requêtes.
        with open(self.fquery,"r") as f:
            balise = None
            idq = None
            for line in f.readlines():
                if(re.search("^[.]I",line)):
                    # Récupération de l'identifiant.
                    idq = int(re.sub("\n","",line.split(" ")[1]))
                    qoll[idq] = Query(idq) 
                    continue 
                if(re.search("^[.][A-Z]",line)):
                    # Pour toute autre balise, initialisation du champ.
                    balise = line[1]
                    continue
                if(line != "\n" and balise == 'W'):
                    # Ne mettre à jour l'objet que s'il s'agit de texte.
                    qoll[idq].addTexte(re.sub("\n"," ",line))  
        # Récupération des documents censés être pertinents.
        with open(self.fperti,"r") as f:
            for line in f.readlines():
                idq, perti = re.findall(r'\d+',line)[0:2]
                # Retrait du 0 en début d'idq pour la correspondance interfichiers
                if idq[0]=='0': idq = idq[1:]
                idq = int(idq)
                qoll[idq].addPertinents(int(perti))
        return qoll
    
    def getQoll(self):
        return self.qoll