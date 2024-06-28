import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# データセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# モデルのトレーニング
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlitのインターフェース
st.title("アヤメの予測アプリ")

st.sidebar.header("パラメータを入力")

# ユーザー入力の取得
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('入力したパラメータ')
st.write(df)

# 予測
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('クラスラベル')
st.write(iris.target_names)

st.subheader('予想')
st.write(iris.target_names[prediction][0])

st.subheader('予想確率')
st.write(prediction_proba)

