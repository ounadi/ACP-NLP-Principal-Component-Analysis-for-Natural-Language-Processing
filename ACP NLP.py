#!/usr/bin/env python
# coding: utf-8

# # TP2 TLN

# ### Nom :Ikram OUNADI
#     

# In[23]:


import spacy
nlp = spacy.load("en_core_web_sm") # Charger le modèle 'en_core_web_sm' (small english model). 
                                   # Dans ce cas, la fonction load() retourne un 'nlp object'
import matplotlib.pyplot as plt


# In[24]:


file1= open("/home/ikram/TLN 2/Text_data_1.txt","r") # Ouvrir le fichier en lecture
file2= open("/home/ikram/TLN 2/Text_data_2.txt","r") # Ouvrir le fichier en lecture

token_list1 = []                 # Créer une liste pour les tokens
for line in file1 :
    one_line1 = nlp(line)        # Faire l'analyse de chaque ligne du fichier via la fonction 'nlp'
    token_list1.extend(one_line1) # Ajouter la ligne analysée à la liste 'token_list' via la fonction 'extend'


# Fermer le fichier
token_list2 = []                 # Créer une liste pour les tokens
for line in file2 :
    one_line2 = nlp(line)        # Faire l'analyse de chaque ligne du fichier via la fonction 'nlp'
    token_list2.extend(one_line2) # Ajouter la ligne analysée à la liste 'token_list' via la fonction 'extend'


# In[25]:


def get_word_vectors(words):
    return [nlp(word).vector for word in words]
    


# In[26]:


list_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS) # Récupérer la liste des stopwords


# In[27]:


ponct = [',','.','!','?','\n',';',':','"','“','”','‘','(',')',"'",'[',']','--','...','/'] # Créer une liste contenant les symboles de ponctuation
list_stopwords.extend(ponct)                                                              # Ajouter les ponctuations à la liste des stopwords


# In[28]:


token_list_filtred1 = [token for token in token_list1 if token.text not in list_stopwords] # Eliminer les tokens correspondant aux stopwords
token_list_filtred2 = [token for token in token_list2 if token.text not in list_stopwords] # Eliminer les tokens correspondant aux stopwords


# In[29]:


token_list_filtred11 = [token for token in token_list_filtred1 # Eliminer les pronoms et les verbes
                       if token.lemma_ !='-PRON-'
                       and token.lemma_ != '-'
                       and token.pos_ != 'VERB'
                      ]
token_list_filtred22 = [token for token in token_list_filtred2 # Eliminer les pronoms et les verbes
                       if token.lemma_ !='-PRON-'
                       and token.lemma_ != '-'
                       and token.pos_ != 'VERB'
                      ]


# In[30]:


token_list_filtred11_lemma = [token.lemma_ for token in token_list_filtred11] # Récupérer tous les lèmmes de la nouvelle liste des tokens
word_counter1 = {}                                                           # Créer un dicionnaire
for word in token_list_filtred11_lemma:                                      # Compter le nombre d'apparition de chaque lèmme
        if word in word_counter1:
            word_counter1[word] += 1
        else:
            word_counter1[word] = 1


# In[31]:


token_list_filtred22_lemma = [token.lemma_ for token in token_list_filtred22] # Récupérer tous les lèmmes de la nouvelle liste des tokens
word_counter2 = {}                                                           # Créer un dicionnaire
for word in token_list_filtred22_lemma:                                      # Compter le nombre d'apparition de chaque lèmme
        if word in word_counter2:
            word_counter2[word] += 1
        else:
            word_counter2[word] = 1


# In[32]:


SortedFiltreLemma=dict();    
list_keys1 = []            
for key, value in sorted(word_counter1.items(), key=lambda item: item[1],reverse=True):
    SortedFiltreLemma[key]=value
    list_keys1.append(key)
SortedFiltreLemma=dict();    
    
list_keys2 = []            
for key, value in sorted(word_counter2.items(), key=lambda item: item[1],reverse=True):
    SortedFiltreLemma[key]=value
    list_keys2.append(key)


# In[48]:


words1=list_keys1[0:40]
print("la liste 1 :" ,words1)

words2=list_keys2[0:40]
print("la liste 2 : " ,words2)


# In[14]:


#appliquer PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
words=words1
pca.fit(get_word_vectors(words1))
word_vecs_2d_1 = pca.transform(get_word_vectors(words1))
word_vecs_2d_1


# In[21]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
words=words2
pca.fit(get_word_vectors(words2))
word_vecs_2d_2 = pca.transform(get_word_vectors(words2))
word_vecs_2d_2


# In[45]:


plt.figure(figsize=(20,5))
#présente les coordonnées de chacun des mots de chacun des deux fichiers
plt.scatter(word_vecs_2d_1[:,0], word_vecs_2d_1[:,1],marker="x",color = 'purple')
plt.scatter(word_vecs_2d_2[:,0], word_vecs_2d_2[:,1],marker="o",color = 'blue')


plt.show()


# In[44]:


plt.figure(figsize=(20,5))
#présente les mots correspondant à chacun des deux fichiers
plt.scatter(word_vecs_2d_1[:,0], word_vecs_2d_1[:,1],color = 'white')
plt.scatter(word_vecs_2d_2[:,0], word_vecs_2d_2[:,1],color = 'white')
for word, coord in zip(words1, word_vecs_2d_1):
    x, y = coord
    plt.text(x, y, word, size= 15,color = 'purple')
for word, coord in zip(words2, word_vecs_2d_2):
    x, y = coord
    plt.text(x, y, word, size= 15,color = 'blue')
plt.show()


# In[ ]:




