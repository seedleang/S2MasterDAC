# -*- coding: utf-8 -*-

import re
import numpy as np
from abc import ABC, abstractmethod

class Document(object):
    def __init__(self,iddoc,details={}):
        # Construction d'un document à partir d'un identifiant 
        # et de quelques métadonnées (champs potentiellement vides).
        self.iddoc = iddoc
        if 'W' in details.keys(): self.texte = details['W']
        else: self.texte = ""
        if 'T' in details.keys(): self.titre = details['T']
        else: self.titre = ""
        if 'B' in details.keys(): self.date = details['B']
        else: self.date = ""
        if 'A' in details.keys(): self.auteur = details['A']
        else: self.auteur = ""
        if 'K' in details.keys(): self.mc = details['K']
        else: self.mc = ""
        if 'X' in details.keys(): self.liens = details['X']
        else: self.liens = []
    
    def snippet(self):
        # Affichage de l'aperçu d'un objet Document.
        print(self.iddoc,"\n",self.titre,"\n",self.texte,"\n")
        
    # Getters et setters - si besoin.
    def getID(self):
        return self.iddoc
    def getTexte(self):
        return self.texte
    def getTitre(self):
        return self.titre
    def getDate(self):
        return self.date
    def getAuteur(self):
        return self.auteur
    def getMc(self):
        return self.mc
    def getLiens(self):
        return self.liens
    
    def addTexte(self,texte):
        self.texte = texte
    def addTitre(self,titre):
        self.titre = titre
    def addDate(self,date):
        self.date = date
    def addAuteur(self,auteur):
        self.auteur = auteur
    def addMc(self,mc):
        self.mc = mc
    def addLiens(self,liens):
        self.liens = [liens]

class Parser(object):
    def __init__(self,file):
        # Chemin du fichier à parser.
        self.file = file
        # Lancement automatique des opérations.
        self.coll = self.parsing()
        
    def parsing(self): 
        # Parsing de l'objet passé en entrée.
        coll = {}
        with open(self.file,"r") as f:
            balise = None
            iddoc = None
            for line in f.readlines():
                if(re.search("^[.]I",line)):
                    # Récupération de l'identifiant.
                    iddoc = int(re.sub("\n","",line.split(" ")[1]))
                    coll[iddoc] = {}
                    continue
                if(re.search("^[.][A-Z]",line)):
                    # Pour toute autre balise, initialisation du champ.
                    balise = line[1]
                    if balise == 'X': coll[iddoc][balise] = []
                    else: coll[iddoc][balise] = ""
                    continue
                if(line != "\n"):
                    # Il s'agit d'une ligne pertinente qui suit une balise.
                    # On conserve soit la liste des numéros cités comme liens,
                    # soit les données textuelles en transformant les sauts de ligne.
                    if balise == 'X': coll[iddoc][balise] += re.findall(r'\d+',line)
                    else: coll[iddoc][balise] += re.sub("\n"," ",line)
        # Création de l'objet Document final avec un peu de nettoyage.
        for document in coll.keys():
            coll[document]['T'] = re.sub(r'\ +'," ",coll[document]['T'].strip())
            if 'W' in coll[document].keys(): coll[document]['W'] = re.sub(r'\ +'," ",coll[document]['W'].strip())
            coll[document] = Document(document,coll[document])
        return coll
    
    # Getter de circonstance.
    def getResult(self):
        return self.coll