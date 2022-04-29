import streamlit as st
import pickle
import pandas as pd
#import sklearn
#loading dataframe
df = pickle.load(open('df.pkl','rb'))
df_user_rating_pivot = pickle.load(open('pivot_table.pkl','rb'))
model_knn = pickle.load(open('model_knn.pkl','rb'))

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


booklist = pickle.load(open('books_list.pkl','rb'))
st.title("Sentimental Books Recommendation")
st.caption('Read only what was destined for you')
selectedbook = st.selectbox(
     'Search for a Book',
     booklist)




if st.button('Recommend'):
    st.write(selectedbook)
summary = df[df['book_name'] == selectedbook]



def recommend(x):
    aa=summary['clean'].iloc[0]
    st.write(aa)
    
    
    a1=[]
    count=0
    for i in range(len(df)):
        count+=1
        a=similar(df['clean'].iloc[i],aa)
        a1.append(a)

    a1 = pd.DataFrame(a1)
    df['similarity'] = a1

    finaldf = df[df['similarity']>0.01]
    #st.write(finaldf)
    
    
    
    for i in range(len(df_user_rating_pivot)):
        if (df_user_rating_pivot.index[i]==selectedbook):
            count=i
            #st.write(count)
            query_index=count
    
    
    
    
    
    distances, indices = model_knn.kneighbors(df_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 16)
    df_user_rating_pivot.index[query_index]


    yash = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            st.write('Recommendations for {0}:\n'.format(df_user_rating_pivot.index[query_index]))
        else:
            yash.append('{0}: {1}, with distance of {2}:'.format(i,df_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i])) 
            #print('{0}: {1}, with distance of {2}:'.format(i,df_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
            #remdup = remdup.append(df[df['book_name'] == df_user_rating_pivot.index[indices.flatten()[i]]])
        #print(df_user_rating_pivot.index[indices.flatten()[i]])
    return st.write(yash)
    
    
    
    
recommend(selectedbook)   
 