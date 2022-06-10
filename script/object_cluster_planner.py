import sys
import math
from tabnanny import check
sys.path.append('../data')

from pantry_database import misc 
from sklearn.preprocessing import StandardScaler
from scipy import spatial
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader

FOOD_GROUP_WEIGHT =  .703
CONTAINER_WEIGHT = .710
STABLE_WEIGHT = .898
PURPOSE_WEIGHT = .908
WORD2VECT_WEIGHT = .7801


def catigory_score(pantryGroup, groceryGroup):
    Pvector = np.array([x for x in pantryGroup.values()])
    Gvector = np.array([x for x in groceryGroup.values()])
    if sum(Pvector) == 0 or sum(Gvector) == 0:
        score = 0
    else:
        score = 1- spatial.distance.cosine(Pvector, Gvector) 
        # Try score with difference similarity functions 
            #spatial.distance.eclidean(Pvector, Gvector) 
            #spatial.distance.jaccard(Pvector, Gvector) 
    return score


def word2vec_score(glove_wiki_vec,pantry,grocery):
    complex_words = [ "gummy-vitamins", "baby-formula", "condensed-milk", "alfredo-sauce","olive-oil", "granola-bars","pancake-mix", "peanut-butter", "grape-jam","bread-crumbs", "hot-sauce", "tomato-soup", "muffin-mix"]
    wd_score = 0 

    if pantry not in complex_words and grocery not in complex_words:
        wd_score += glove_wiki_vec.similarity(pantry, grocery)
    else:
        if pantry in complex_words and grocery in complex_words:
            pantry = pantry.split("-")
            grocery = grocery.split("-")
            pantry = glove_wiki_vec[pantry[0]] +  glove_wiki_vec[pantry[1]]
            grocery = glove_wiki_vec[grocery[0]] +  glove_wiki_vec[grocery[1]]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
        elif pantry in complex_words:
            pantry = pantry.split("-")
            pantry = glove_wiki_vec[pantry[0]] +  glove_wiki_vec[pantry[1]]
            grocery = glove_wiki_vec[grocery]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
        elif grocery in complex_words:
            grocery = grocery.split("-")
            grocery = glove_wiki_vec[grocery[0]] +  glove_wiki_vec[grocery[1]]
            pantry = glove_wiki_vec[pantry]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
    
    return wd_score

def cluster(pantry_items, grocery_items):
    """
    Return: 
        { grocy_item: [(pantry_item, score), (pantry_item,score)], grocery....}
    """
    glove_wiki_vec = gensim.downloader.load('glove-wiki-gigaword-300')
    clusters = {}
    for grocery in grocery_items:
        score_list = []
        for pantry_obj in pantry_items:

            fg_score = FOOD_GROUP_WEIGHT *  catigory_score(misc[pantry_obj]["Food_Groups"], misc[grocery]["Food_Groups"])
            continer_score = CONTAINER_WEIGHT *  catigory_score(misc[pantry_obj]["Continer"], misc[grocery]["Continer"])

            stability_score = STABLE_WEIGHT * catigory_score(misc[pantry_obj]["Stability"], misc[grocery]["Stability"])
            purpose_score = PURPOSE_WEIGHT *  catigory_score(misc[pantry_obj]["Purpose"], misc[grocery]["Purpose"])
            #sum(np.array([x for x in misc[pantry_obj]["Purpose"].values()]) *
            #             np.array([x for x in misc[grocery]["Purpose"].values()]))

            wdVec_score = WORD2VECT_WEIGHT * word2vec_score(glove_wiki_vec ,pantry_obj,grocery)

            score = fg_score + continer_score + stability_score + purpose_score + wdVec_score
            score_list.append(( pantry_obj, score))
            # if pantry_obj in self.shelf1:
            #     score_list.append(( pantry_obj, score, self.shelf1))
            # elif pantry_obj in self.shelf2:
            #     score_list.append(( pantry_obj, score, self.shelf2))
                
        
        score_list.sort( key = lambda x: x[1], reverse=True)
        clusters[grocery] = score_list

    #self.clustered = clusters      
    return clusters



class Grocery_cluster:
    def __init__(self,shelf1 =None,shelf2 = None, pantrys= None, grocerys=None):
        if shelf1 != None:
            self.shelf1 = shelf1 # list ( [ p1,p3,p9...], height)
            self.shelf2 = shelf2      
            self.pantry_items = pantrys 
            self.grocery_items = grocerys
        else:
            self.shelf1 =  [ "cookies", "pringles", "granola-bars"]
            self.shelf2 = ["spam", "muffin-mix","mustard","beans"]      
            self.pantry_items = [ "peanuts","cookies", "grape-jam", "spam", "apple", "onion","sugar", "bread", "ketchup"]
            self.grocery_items = ["tomato-soup", "hot-sauce", "pancake-mix", "peanut-butter"]
        self.clustered = None
        self.glove_wiki_vec = gensim.downloader.load('glove-wiki-gigaword-300')

    #def cluster(pantry_items:'list',grocery_items:'list')-> 'dict':
    def cluster(self):
        """
        Return: 
            { grocy_item: [(pantry_item, score), (pantry_item,score)], grocery....}
        """
        clusters = {}
        for grocery in self.grocery_items:
            score_list = []
            for pantry_obj in self.pantry_items:

                fg_score = FOOD_GROUP_WEIGHT * catigory_score(misc[pantry_obj]["Food_Groups"], misc[grocery]["Food_Groups"])
                continer_score = CONTAINER_WEIGHT * catigory_score(misc[pantry_obj]["Continer"], misc[grocery]["Continer"])

                stability_score = STABLE_WEIGHT * catigory_score(misc[pantry_obj]["Stability"], misc[grocery]["Stability"])
                purpose_score = PURPOSE_WEIGHT * catigory_score(misc[pantry_obj]["Purpose"], misc[grocery]["Purpose"])
                
                wdVec_score = WORD2VECT_WEIGHT * word2vec_score(self.glove_wiki_vec, pantry_obj,grocery)

                score = fg_score + continer_score + stability_score + purpose_score + wdVec_score
                if pantry_obj in self.shelf1[0]:
                    score_list.append(( pantry_obj, score, self.shelf1[1]))
                elif pantry_obj in self.shelf2[0]:
                    score_list.append(( pantry_obj, score, self.shelf2[1]))
                    
                
            score_list.sort( key = lambda x: x[1], reverse=True)
            clusters[grocery] = score_list

        self.clustered = clusters      
        return clusters
    



