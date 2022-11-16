#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")

model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")

##### Process text sample (from wikipedia)


# In[2]:


import pandas as pd
import csv
from transformers import pipeline
def isNaN(string):
    return string != string
extractionEntity = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
df = pd.read_csv('data.csv')


# In[3]:


df


# In[4]:


vraiPositif=0
counterSpecialCara = 0
extrait = 0
numDateNesInData = 0
numTextInData = 0
fauxPositif = 0

wordList = ["le", "un", "en"]
with open('newResult.csv', 'w', newline='', encoding="utf8") as file:
    writer = csv.writer(file, delimiter =';')
    writer.writerow(["index", "text", "extraite", "score", "annotée"])
    for i in df.index:
        if not isNaN(df['dateNes'][i]):
            numDateNesInData+=1
        if not isNaN(df['texte'][i]):
            numTextInData+=1
            entities = extractionEntity(df['texte'][i])
            for entity in entities :
                if (entity['entity_group'] == "DATE") :
                    #if ( not isNaN(df['dateNes'][i])) :
                        extrait+=1
                        if any(word in entity['word'] for word in wordList): 
                            counterSpecialCara+=1
                        #print(i, "text = ",df['texte'][i], "|||| extraite =", entity['word'] ,"|||| score = ", entity['score'] ,"|||| annotée =", df['dateNes'][i])
                        writer.writerow([i,df['texte'][i], entity['word'], entity['score'], df['dateNes'][i]])
                        
    file.close()


# In[5]:


numDateNesInData


# In[6]:


numTextInData


# In[7]:


extrait


# In[8]:


counterSpecialCara


# In[9]:


with open('newResult.csv', 'r', encoding="utf8") as file:
    
    csv_reader = csv.reader(file, delimiter=';')
    uniqueIds = set()

    for row in csv_reader:
        if isNaN(row[4]) or row[4]=="nan":
            fauxPositif+=1
        #if not isNaN(row[4]) or row[4]!="nan":
        if row[4]!="nan":
            uniqueIds.add(row[0])

vraiPositif = len(uniqueIds)


# In[10]:


repetation = extrait - vraiPositif - fauxPositif


# In[11]:


repetation


# In[12]:


fauxPositif


# In[13]:


vraiPositif


# In[14]:


fauxNegatif = numDateNesInData - vraiPositif


# In[15]:


fauxNegatif


# In[16]:


vraiNegatif = numTextInData-numDateNesInData-fauxPositif


# In[17]:


vraiNegatif


# In[18]:


precision = vraiPositif/(vraiPositif+fauxPositif)


# In[19]:


precision


# In[20]:


rappel =  vraiPositif/(vraiPositif+fauxNegatif)


# In[21]:


rappel


# In[22]:


f1score=(2*precision*rappel)/(precision+rappel)


# In[23]:


f1score


# In[24]:


Accuracy=(vraiNegatif +vraiPositif)/(vraiNegatif+fauxPositif+vraiPositif+fauxNegatif)


# In[25]:


Accuracy


# In[26]:


with open('newStat.csv', 'w', newline='', encoding="utf8") as file:
    writer = csv.writer(file, delimiter =';')
    writer.writerow(["originalFileDates","fauxNegatif", "vraiPositif", "fauxPositif","vraiNegatif", "precision","rappel", "f1score","Accuracy"])
    writer.writerow([numDateNesInData,fauxNegatif, vraiPositif, fauxPositif, vraiNegatif, precision, rappel, f1score, Accuracy])
    file.close()


# In[27]:


read = pd.read_csv('newStat.csv')


# In[28]:


read


# In[29]:


print("Number data in orginal file  = " , numTextInData)
print("Number dates existe in original file = " , numDateNesInData)
print("Number de data extrait avec camembert = " ,extrait )
print("Number des dates afficher sans repetation et sans fauxpositif => vraipositif = " , vraiPositif)
print("Nombre de repetation existe dans l'extrait = " , repetation)
print("Number des adresses existe dans l'extait = ",fauxPositif)
print("Number des dates no sort dans le resultat = ",fauxNegatif)
print("Number des data no sort dans le resultat = ",vraiNegatif)
print ("Number des dates extratit avec special caracters = ", counterSpecialCara)


# In[ ]:




