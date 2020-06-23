# -*- coding: utf-8 -*-

import numpy as np
import TextRepresenter as tr

class IndexerSimple(object):
    def __init__(self,coll):
        # Un indexeur concerne une collection,
        self.coll = coll
        self.length = len(self.coll)
        # et sert à la représenter
        self.inds = {}
        self.inv = {}
        self.invnorm = {}
        # avec ou sans scores tf-idf.
        self.doctfi = {}
        self.stemtfi = {}  
        # Lancer automatiquement les opérations
        # à l'évocation de la classe Indexeur.
        self.indexation()
        self.width = len(self.inv)
        self.total = sum([sum(self.inv[s].values()) for s in self.inv.keys()])
        
    def invertThis(self,origidx):
        # Construire un dictionnaire 
        # pour la collection entière.
        lexicon = {}
        inverted = {}
        for newwords in origidx.values():
            lexicon.update(newwords)
        # Pour chacun des mots cités,
        for stem in lexicon:
            # détailler une entrée dans l'index inverse,
            inverted[stem] = {}
            # et lister les documents concernés.
            for source in origidx.keys():
                if stem in origidx[source]:
                    inverted[stem][source] = origidx[source][stem]
        return inverted
                    
    def normaliseThis(self,idx):
        normalised = {}
        # S'assurer que chaque sous-dictionnaire
        # de l'index considéré (inverse ou non) somme à 1
        # - représentation en proportion des composantes des documents
        # - représentation en proportion des documents citant une racine
        for anykeytype in idx.keys():
            fact = sum(idx[anykeytype].values())
            normvals = map(lambda o: o/fact, list(idx[anykeytype].values()))
            normalised[anykeytype] = dict(zip(list(idx[anykeytype].keys()),normvals))
        return normalised
    
    def computeIdf(self,stem):
        # L'IDF d'un mot est constante pour tout le corpus.
        if stem in self.inv.keys(): return np.log((1+self.length)/(1+len(self.inv[stem])))
        else: return np.log((1+self.length))
    
    def computeTfidf(self):
        scored = {}
        for source in self.inds.keys():
            scored[source] = {}
            for stem, o in self.inds[source].items():
                # Faire en sorte que des mots omniprésents dans le corpus
                # soient moins discriminants pour l'identité d'un document.
                scored[source][stem] = o*self.computeIdf(stem)
        return scored  
        
    def indexation(self):
        # Outil de représentation du texte
        # pour analyse fréquentielle.
        p = tr.PorterStemmer()
        # Parcourir la collection,
        for iddoc in self.coll.keys():
            source = self.coll[iddoc]
            # et obtenir le découpage de chaque ressource textuelle.
            self.inds[iddoc] = p.getTextRepresentation(source.getTexte()+source.getTitre())
        # A partir de l'index, il est facile d'obtenir l'index inverse,
        self.inv = self.invertThis(self.inds)
        # puis de le normaliser.
        self.invnorm = self.normaliseThis(self.inv)
        # Pareil pour les index "pondérés".
        self.doctfi = self.computeTfidf()
        self.stemtfi = self.invertThis(self.doctfi) 
        
    # Getters utiles.
    def getSize(self):
        return [self.length,self.width,self.total]
    def getIndex(self):
        return self.inds
    def getInverted(self):
        return dict(sorted(self.inv.items()))
    def getInvertedN(self):
        return dict(sorted(self.invnorm.items()))
        
    # Getters requis.
    def getTfsForDoc(self,iddoc):
        return dict(sorted(self.inds[iddoc].items())) if iddoc in self.inds.keys() else {}
    def getTfsForStem(self,stem):
        return self.inv[stem] if stem in self.inv.keys() else {}
    def getTfIDFsForDoc(self,iddoc):
        return dict(sorted(self.doctfi[iddoc].items())) if iddoc in self.doctfi.keys() else {}
    def getTfIDFsForStem(self,stem):
        return self.stemtfi[stem] if stem in self.stemtfi.keys() else {}
    def getStrDoc(self,iddoc):
        return self.coll[iddoc].texte if iddoc in self.coll.keys() else None