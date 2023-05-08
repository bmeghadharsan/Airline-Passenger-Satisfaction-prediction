#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
pickle_in = open("data/DT.pkl","rb")
classifier = pickle.load(pickle_in)
pickle_in1 = open("data/LogReg.pkl","rb")
logreg = pickle.load(pickle_in1)
pickle_in2 = open("data/random_forest_classifier.pkl","rb")
rf = pickle.load(pickle_in2)


# In[15]:


import pandas as pd
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/pp1_test.csv')
X = df.drop(columns = ["satisfaction"],axis = 1)
y = df["satisfaction"]
features = tuple(df.columns)
train_df = pd.read_csv("data/pp1_train.csv")
X1 = train_df.drop(columns = ["satisfaction"])
y1 = train_df["satisfaction"]


# In[16]:


from sklearn.metrics import confusion_matrix


# In[17]:


def predict_note_authentication(opt,pred):
    prediction=None
    if opt=='Decision Tree':
        prediction = classifier.predict([pred])
    if opt=='Logistic Regression':
        prediction = logreg.predict([pred])
    if opt=='Random Forest':
        prediction = rf.predict([pred])
    return prediction


# In[18]:


# In[ ]:


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")
animation_symbol = "❄"
animation_symbol1 = "✈"
st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol1}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol1}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol1}</div>
    """,
    unsafe_allow_html=True,
)


# In[ ]:


from streamlit_lottie import st_lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def lottie(url):
    lottie_hello = load_lottieurl(url)

    st_lottie(
        lottie_hello,
        height=400,)

# In[19]:

st.title("Airline Passenger Satisfaction")
st.subheader("Dataset")
if st.button("View"):
    st.dataframe(df)
choice=st.selectbox('Select your Choice',('Vanakkam','Plots','Models','Customize Models'))

#Models
if choice=='Models':
    option = st.selectbox('Choose the Model: ',
        ('Select Model','Decision Tree', 'Logistic Regression', 'Random Forest'))
    st.write('You selected:',option)
    if st.button("Calculate Accuracy"):
        if option=='Decision Tree':
            y_pred = classifier.predict(X)
            
        if option=='Logistic Regression':
            y_pred = logreg.predict(X)
            
        if option=='Random Forest':
            y_pred = rf.predict(X)
            
        st.success(f"Test Accuracy: {accuracy_score(y,y_pred)}")
        st.progress(accuracy_score(y,y_pred))
        st.write(f"{accuracy_score(y,y_pred)*100}%")
        
        conf_matrix = confusion_matrix(y_pred,y)
        fig,ax = plt.subplots()
        sns.heatmap(conf_matrix,annot = True,fmt = "")
        plt.ylabel("Actual Positive                 Actual Negative")
        plt.xlabel("Predicted Negative                 Predicted Positive")
        st.write(fig)
    if st.button("Do you want to predict Output for your inputs with this Model?"):
        df1=pd.read_csv("data/inputs.csv")
        predictions=df1.iloc[0].to_list()
        result = predict_note_authentication(option,predictions)
        if result[0]==0:
            st.success(f"Output: 'Neutral or Not Satisfied'")
        else:
            st.success(f"Output: 'Satisfied'")
        st.write("Note: Work in Progress on Upload Option!!, You have to store your inputs.csv in the path of run file")
        st.write("Inputs from 'inputs.csv:'")
        st.write(predictions)
        
if choice=='Vanakkam':
    lottie("https://assets7.lottiefiles.com/packages/lf20_DXljHQsLLA.json")
        
#Plots
if choice=='Plots':
    st.subheader("Correlations")
    if st.button("View Correlations"):
        fig, ax = plt.subplots()
        plt.figure(figsize = (25,25))
        sns.heatmap(df.corr(), ax=ax)
        st.write(fig)
    st.subheader("Calculate Correlations")
    st.text("Feature 1")
    option_f1 = st.selectbox("Enter feature1",features)
    st.text("Feature 2")
    option_f2 = st.selectbox("Enter feature2",features)

    if st.button("Calculate Correlation"):
        st.success(f"Correlation between {option_f1} and {option_f2} is {df.corr()[option_f1][option_f2]}")
        fig, ax = plt.subplots()

        sns.scatterplot(option_f1,option_f2,data = df)
        st.write(fig)
    st.subheader("Data Visualization")
    option_feature = st.selectbox("Enter feature for visualization",features)

    st.text("Suitable Plots for this feature")
    if len(df[option_feature].unique()) > 8:
        option_plot = st.selectbox("Select your plot",("Histogram","Dist Plot"))
    else:
        option_plot = st.selectbox("Select your plot",("Count plot","Pie Chart"))

    if option_plot == "Histogram":
        fig,ax = plt.subplots()
        sns.histplot(df[option_feature],bins = 30)
    elif option_plot == "Dist Plot":
        fig,ax = plt.subplots()
        sns.distplot(df[option_feature],bins = 30)
    elif option_plot == "Count plot":
        fig,ax = plt.subplots()
        sns.countplot(df[option_feature])
    elif option_plot == "Pie Chart":
        fig,ax = plt.subplots()
        df.groupby(option_feature).size().plot(kind = "pie",y="",autopct = '%1.0f%%')
        plt.legend()
    st.write(fig)

#Customize Models
if choice=="Customize Models":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    st.header("Customize your model")
    option_custom = st.selectbox(
    "Select Model",("Decision Tree Classifier","Logistic Regression","SVM","KNN"))
    
    if option_custom == "Decision Tree Classifier":
        st.subheader("Parameters")
        max_d = st.number_input(label = "Max depth",min_value = 3,max_value = 50)
        cr = st.selectbox("Select Criteria",("entropy","gini"))
        dc = DecisionTreeClassifier(max_depth = max_d,criterion = cr)

        if st.button("Train your model"):
            dc.fit(X1,y1)
            st.success(f"Model Accuracy {accuracy_score(y,dc.predict(X))}")
            
    elif option_custom == "SVM":
        st.subheader("Parameters")
        ker = st.selectbox("Kernel",("linear","poly","sigmoid","rbf"))
        gamma = st.selectbox("Gamma",("scale","auto"))
        sv_model = SVC(kernel = ker,gamma = gamma)
        if st.button("Train your model"):
            sv_model.fit(X1,y1)
            st.success(f"Model Accuracy {accuracy_score(y,sv_model.predict(X))}")
            
    elif option_custom == "Logistic Regression":
        st.subheader("Parameters")
        pen = st.selectbox("Penalty",("l1","l2","elasticnet"))
        solver = st.selectbox("Solver",("lbfgs","liblinear","newton-cg","sag","saga"))
        c=  st.number_input(label = "C",min_value = 1,max_value = 50)
        lr = LogisticRegression(C = c,penalty = pen,solver = solver)
        if st.button("Train your model"):
            lr.fit(X1,y1)
            st.success(f"Model Accuracy {accuracy_score(y,lr.predict(X))}")
            
    elif option_custom == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        st.subheader("Parameters")
        st.text("n_neighbors")
        nei = st.number_input(label = "",min_value = 1, max_value = 100)
        knn = KNeighborsClassifier(n_neighbors = nei)
        if st.button("Train your model"):
            knn.fit(X1,y1)
            st.success(f"Model Accuracy {accuracy_score(y,knn.predict(X))}")
    
 

