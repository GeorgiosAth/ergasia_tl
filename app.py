
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="DataDive", page_icon=":bar_chart:",layout="wide")

st.title(" Transforming spreadsheets into insights")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        return data
    return None

def plot_2d(data, method='PCA'):
    if method == 'PCA':
        pca = PCA(n_components=2)
        components = pca.fit_transform(data.iloc[:, :-1])
    else:
        tsne = TSNE(n_components=2)
        components = tsne.fit_transform(data.iloc[:, :-1])

    fig, ax = plt.subplots()
    scatter = ax.scatter(components[:, 0], components[:, 1], c=data.iloc[:, -1], cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    st.pyplot(fig)

def eda_plots(data):
    st.subheader("Exploratory Data Analysis")
    st.bar_chart(data.describe())
    st.line_chart(data.corr())
    st.area_chart(data.isnull().sum())

def classification(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.subheader("Classification")

    k = st.slider("Select k for k-NN", 1, 15, 3)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    st.write(f"k-NN Accuracy: {acc_knn}")

    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    st.write(f"SVM Accuracy: {acc_svm}")

def clustering(data):
    X = data.iloc[:, :-1]

    st.subheader("Clustering")

    k = st.slider("Select k for k-means", 1, 15, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)
    silhouette_kmeans = silhouette_score(X, labels_kmeans)

    st.write(f"k-means Silhouette Score: {silhouette_kmeans}")

    eps = st.slider("Select eps for DBSCAN", 0.1, 5.0, 0.5)
    dbscan = DBSCAN(eps=eps)
    labels_dbscan = dbscan.fit_predict(X)
    silhouette_dbscan = silhouette_score(X, labels_dbscan) if len(set(labels_dbscan)) > 1 else -1

    st.write(f"DBSCAN Silhouette Score: {silhouette_dbscan}")

st.sidebar.title("Info")
st.sidebar.info("""
This application is developed for data mining and analysis using Streamlit.
- **Team Members:**
  - Member 1: Task A
  - Member 2: Task B
  - Member 3: Task C
- **Instructions:**
  - Load your dataset in CSV or Excel format.
  - Use the tabs to visualize data and apply machine learning algorithms.
""")

data = load_data()
if data is not None:
    st.write(data)

    tab1, tab2, tab3, tab4 = st.tabs(["2D Visualization", "EDA", "Classification", "Clustering"])

    with tab1:
        st.subheader("2D Visualization")
        method = st.selectbox("Select Dimensionality Reduction Method", ["PCA", "t-SNE"])
        plot_2d(data, method)

    with tab2:
        eda_plots(data)

    with tab3:
        classification(data)

    with tab4:
        clustering(data)

