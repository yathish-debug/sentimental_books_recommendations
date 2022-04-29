import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
import pickle
from flask import Flask


df = pd.read_csv('newdfnlp.csv')
df2 = df['book_name'].drop_duplicates()
bookname=[i for i in df2]


from tkinter import *
root = Tk()
root.geometry( "200x200" )

def show():
    label.config( text = clicked.get() )
  
clicked = StringVar()

clicked.set( "Search for a book" )

drop = OptionMenu( root , clicked , *bookname )
drop.pack()
  
# Create button, it will change label text
button = Button( root , text = "Get recommendation" , command = show ).pack()
  
# Create Label
label = Label( root , text = " " )
label.pack()

root.mainloop()



df1 = df[['User-ID', 'book_name', 'Book-Rating','genre_new']]
df1 = df1.drop_duplicates(['User-ID', 'book_name'])
pickle.dump(df,open('df.pkl','wb'))

import pandas as pd
from scipy.sparse import csr_matrix
df1.set_index(['book_name', 'User-ID']).unstack()
df_user_rating_pivot = df1.pivot(index = 'book_name', columns = 'User-ID', values = 'Book-Rating').fillna(0)
df_user_rating_matrix = csr_matrix(df_user_rating_pivot.values)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(df_user_rating_matrix)
df_user_rating_pivot
pickle.dump(df_user_rating_pivot,open('pivot_table.pkl','wb'))
pickle.dump(model_knn,open('model_knn.pkl','wb'))

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


from tqdm import tqdm

selectedbook = clicked.get()
summary = df[df['book_name'] == selectedbook]#.clean_summary
remdup = summary.drop_duplicates(['book_name'])
remdup = pd.DataFrame(remdup)
print(selectedbook)
print(remdup)
#aa = remdup.clean_summary()

def recommend(x):
    aa=summary['clean'].iloc[0]
    a1=[]
    count=0
    for i in range(len(df)):
    
        count+=1
        a=similar(df['clean'].iloc[i],aa)
        a1.append(a)

    a1 = pd.DataFrame(a1)
    df['similarity'] = a1

    finaldf = df[df['similarity']>0.01]

    for i in range(len(df_user_rating_pivot)):
        if (df_user_rating_pivot.index[i]==selectedbook):
            count=i
            print(count)
            query_index=count


    distances, indices = model_knn.kneighbors(df_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 16)

    df_user_rating_pivot.index[query_index]


    yash = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(df_user_rating_pivot.index[query_index]))
        else:
            yash.append('{0}: {1}, with distance of {2}:'.format(i,df_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i])) 
            #print('{0}: {1}, with distance of {2}:'.format(i,df_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
            #remdup = remdup.append(df[df['book_name'] == df_user_rating_pivot.index[indices.flatten()[i]]])
        #print(df_user_rating_pivot.index[indices.flatten()[i]])
    return yash
print(recommend(summary))




pickle.dump(bookname,open('books_list.pkl','wb'))














