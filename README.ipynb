import re
import os
import sys

import pandas as pd
import numpy as np
import langid
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from spacy_langdetect import LanguageDetector
from nltk.classify.textcat import TextCat
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import unicodedata


#add_stop_hs84 = ['', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\',
  #          'baik', 'dan', 'baru', ' x ', ' ii', ' mm',
  #          ' kg', ' kgm','ml','red',' l'
  #         'pcs', 'promotional','promotion','free of charge','ml','foc']


'''
For the data sets under consideration we perform a number of steps to preprocess
the data. By means of a regular expression, we 
first obtain a format
of one description per line. We then replace diacritics with their unaccented
form and convert the text to lowercase UTF-8.
We preserve hyphens, but otherwise have a rather strict removal policy: we
only keep alpha-numeric content, including spaces. However, extra white
space is also removed, as words will be tokenized using the remaining spaces.
However, we do not correct for spelling mistakes (such as dinning instead of
dining) which are frequent with our human-entered data. We do also not remove
stop words, or employ a stemming or lemmatization strategy. Lastly,
we do not group common phrases together to make them appear in the data
as one token (such as "new york" instead of "new york"). We also do not
perform any part-of-speech (POS) tagging, syntactic dependency parsing,
or named entity recognition (NER). -- Luppes, 2019
'''


'''
Data pre-processing is carried out after cleaning, which includes: removal of numbers, punctuations, and contents
within brackets; converting all words into upper case; deleting stop-words using a standard list; and removal of data
duplications for each category. Strings with joint alphabetical and non-alphabetical characters are segmented and
treated separately, and records with missing attributes are removed.
After all the above steps, the removal of noisy data is carried out. Here noisy data refers to the records with the
same product description yet appearing under multiple categories with different HS codes.
-- Liya Ding et al, 2015
'''

factory = StopWordRemoverFactory()
swindo = factory.get_stop_words()



nlp = spacy.load("en")
nlp.Defaults.stop_words 

#nlp.Defaults.stop_words -= nlp.Defaults.stop_words ## ini untuk nonaktifkan stopword bawaan spacy dan pake custom list 
#swcustom = open("D:\ML\hspred\hs01\stopwords_hs01.txt", "r").read().split()
#n#####ew_stopwords = stopwords.union(new_words)
#nlp.Defaults.stop_words |= set(new_stop_ur01)
#nlp.Defaults.stop_words

''' Hard-coded Version : '''

nlp.Defaults.stop_words |= {"semoga","pegawai","baik","semua","kementerian", "keuangan", "kementerian", "cukup", "lebih"
                            ,"baik", "terkait", "sangat","bagu","bagus", "sangat","terima","kasih", "mantap", "lanjutkan", "oke",
                            "saran", "sesuai", "tingkatkan","dan","yang","di","ok", "tidak", "ada"
                           }

swcustom = nlp.Defaults.stop_words 

def process_text(text):
    lowercase = str(text).lower()
    nospecial = re.sub(r'[^\w ]+', " ", lowercase)
    #no_num = re.sub('[0-9]+', '', nospecial)
    doublespace = ' '.join(nospecial.split())
    unaccent = unicodedata.normalize('NFKD', doublespace).encode('ascii', 'ignore').decode('utf-8', 'ignore')   
    #no_stop = [t for t in unaccent if t not in stopwords]
    notabs = re.sub(r'^\s*|\s\s*', ' ', unaccent).strip()
    #no_stop = ' '.join([t for t in notabs.split() if t not in stopwords])
    return notabs
    
def process_all(text): 
    lowercase = str(text).lower()   
    nospecial = re.sub(r'[^\w% ]+', " ", lowercase)
 
    pattern = r'[0-9]'  #percentage di keep
   # pattern = r'\b([0-9]*\d[0-9]*){4,}\b'  #percentage di keep buat no num special
   # no_num = re.sub(pattern, '', nospecial, flags=re.IGNORECASE) #dimodifikasi untuk hapus numeric > 3digit     
    no_num = re.sub(pattern, '', nospecial)
    
    unaccent = unicodedata.normalize('NFKD', no_num).encode('ascii', 'ignore').decode('utf-8', 'ignore')   
    #no_stop = [t for t in unaccent if t not in stopwords]
    notabs = re.sub(r'^\s*|\s\s*', ' ', unaccent).strip()
    #no_stop = ' '.join([t for t in notabs.split() if t not in stopwords])
    notabss = ""
    for word in notabs.split(' '):
        if any(char.isdigit() for char in word) and any(c.isalpha() for c in word):
            #new_s += ''.join([i for i in word if not i.isdigit()])
            notabss +=  word.replace(word,'')       
        else:
            notabss += word
        notabss += ' '
    doublespace = ' '.join(notabss.split())   
    return doublespace  
    
    
def no_sw_en(text): 
    no_stop = ' '.join([t for t in text.split() if t not in stopwords])
    return no_stop  

def no_sw_id(text): 
    no_stop = ' '.join([t for t in text.split() if t not in swindo])
    return no_stop  

def no_sw_custom(text): 
    no_stop = ' '.join([t for t in text.split() if t not in swcustom])
    return no_stop 
    
def count_swen1(text) :
    text1 = remove_first_end_spaces(text)
    text2 = remove_all_extra_spaces(text1)
    return len([t for t in text2.split() if t in stopwords])    
    
def count_swen(text):
    text.split()
    return len([t for t in text.split() if t in stopwords])
 
    
def count_swid(text):
    text.split()
    return len([t for t in text.split() if t in swindo])
 


def set_df(mydataframe):
    for columns in mydataframe.columns:
        mydataframe[columns] = mydataframe[columns].astype(str).str.lower()
    mydataframe.columns= mydataframe.columns.str.lower()
   # d = [col for col in mydataframe.columns if 'hs' or 'hscode' in col]
    d = [col for col in mydataframe.columns if 'hscode' in col]
    #d =d.astype('object')
    #d = np.array(d, dtype=np.object)
    mydataframe['hs8'] = mydataframe[d]
    mydataframe['hs8'] = mydataframe['hs8'].str[:8]
    return mydataframe
  
def load_data(data):
    if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls') is True:
          data = pd.read_excel(filename, dtype=object) 
    elif filename.lower().endswith('.txt') or filename.lower().endswith('.csv') is True:
          data = pd.read_csv(filename, dtype=object,sep="|")
    elif filename.lower().endswith('.json') is True:
          data = pd.read_json(filename, dtype=object)
    return data

#data['ur_clean'] = data['urbrg'].apply(process_text) 


def remove_end_spaces(string):
    return "".join(string.rstrip())


# Remove the first and end spaces
def remove_first_end_spaces(string):
    return "".join(string.rstrip().lstrip())


# Remove all spaces
def remove_all_spaces(string):
    return "".join(string.split())

# Remove all extra spaces
def remove_all_extra_spaces(string):
    return " ".join(string.split())   
    

def jumlahkata(text) :
    text1 = remove_first_end_spaces(text)
    text2 = remove_all_extra_spaces(text1)
    return len(str(text2).split(' '))

def frek_kata(clean_text_list, top_n):
    flat = [item for sublist in clean_text_list for item in sublist]
    with_counts = Counter(flat)
   # with_counts = 20
    top = with_counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T


def jum_karakter(text):
    nospecial = re.sub(r'[^\w ]+', " ", text)
    i = nospecial.split()
    nospecial = ''.join(i)
    return len(nospecial)

def jum_num(text):
    nospecial = re.sub(r'[^\w ]+', " ", text)
    no_num = re.sub('[a-zA-Z]+', '', nospecial)
    doublespace = ' '.join(no_num.split())
    unaccent = unicodedata.normalize('NFKD', doublespace).encode('ascii', 'ignore').decode('utf-8', 'ignore')   
    notabs = re.sub(r'^\s*|\s\s*', ' ', unaccent).strip()
    i = notabs.split()
    notabs = ''.join(i)
    return len(notabs)

vowel = ['a','e','i','o','u']

def remove_novowels(text):
    stop2 = re.findall(r'\b[^AEIOU_0-9\W]+\b', text, flags=re.I)
    novowels = ' '.join([t for t in text.split() if t not in stop2])   
    return novowels
    

def remove_dobelvowels(text):
    stop3 = re.findall(r'\b[^0-9bcdfghjklmnpqrstvwxyz\W]+\b', text, flags=re.I)
    doblevow = ' '.join([t for t in text.split() if t not in stop3])   
    return doblevow

  
def remove_novowelsv2(text): # ini versi ada pembatasan untuk akomodir karakter sperti PVC
    stop2 = re.findall(r'\b[^AEIOU\W]{5,100}\b', text, flags=re.I)
    doblevow = ' '.join([t for t in text.split() if t not in stop2])
    return doblevow
    
def removenummix(text):
    aa = ' '.join(w for w in text.split() if not any(x.isdigit() for x in w))
    return aa

#df['text'].str.findall('\w{4,}').str.join(' ')
    
#text = " space       adadeh   "
#aa = jumlahkata(text)
#aa
    
#tes= "110    HEAD FEEDER  OF  AND   @@@@%#   CATTLE( SAPI BAKALAN / BETINA )"

#jum_karakter(tes)  

#text= "aaa aa ga ggwpjghsggdfgd pp pvd xkcd "
#bersih(text)
