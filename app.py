import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Dashboard Clustering Gempa", layout="wide")
st.title("🌍 Dashboard Analisis Clustering K-Means Gempa Bumi")

@st.cache_data
def load_data():
    return pd.read_csv('global_earthquakes_seismic.csv')

try:
    df = load_data()
    st.subheader("Cuplikan Dataset")
    st.dataframe(df.head())

    # 1. Preprocessing
    num_cols = ['latitude', 'longitude', 'depth', 'mag', 'sig', 'rms', 'gap']
    df_num = df[num_cols].dropna().drop_duplicates()
    
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_num[~((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)].copy()

    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Sidebar untuk parameter interaktif
    st.sidebar.header("Pengaturan K-Means")
    k_clusters = st.sidebar.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=7, value=3)

    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, df_clean['cluster'])
    
    # 4. Menampilkan Metrik
    st.subheader("Metrik Evaluasi")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Silhouette Score (K terpilih)", value=f"{sil_score:.4f}")
    with col_m2:
        st.metric(label="Jumlah Data Bersih (Setelah Outlier Removal)", value=f"{df_clean.shape[0]} baris")
    
    st.subheader("Rata-rata Karakteristik per Cluster")
    st.dataframe(df_clean.groupby('cluster').mean())

    # 5. Visualisasi Utama Berdasarkan K Terpilih (Total 9-10 Grafik)
    st.markdown("---")
    st.header("📊 Visualisasi Hasil Analisis & Clustering")

    # Baris 1: 2D Scatter Plot & Correlation Heatmap
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. 2D Scatter Plot (Longitude vs Latitude)")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df_clean, palette='viridis', s=15, alpha=0.8, ax=ax_scatter)
        ax_scatter.set_title("Geospatial Distribution")
        st.pyplot(fig_scatter)

    with col2:
        st.subheader("2. Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_clean.drop('cluster', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
        st.pyplot(fig_heat)

    # Baris 2: Proyeksi PCA 2D & Geospatial Distribution (Set1)
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Proyeksi PCA 2D")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['pca_1', 'pca_2'])
        df_pca['cluster'] = df_clean['cluster'].values

        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x='pca_1', y='pca_2', hue='cluster', palette='viridis', alpha=0.6, ax=ax_pca)
        ax_pca.set_title('K-Means Clustering (PCA 2D Projection)')
        st.pyplot(fig_pca)

    with col4:
        st.subheader("4. Geospatial Distribution (Palette Set1)")
        fig_geo, ax_geo = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_clean, x='longitude', y='latitude', hue='cluster', palette='Set1', s=15, alpha=0.7, ax=ax_geo)
        ax_geo.set_title('Geospatial Distribution of Earthquake Clusters')
        st.pyplot(fig_geo)

    # Baris 3: Boxplots Gabungan (Mag & Depth) & 3D Scatter Plot
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("5. Distribusi Fitur (Boxplots Gabungan)")
        fig_box, axes_box = plt.subplots(1, 2, figsize=(8, 6))
        sns.boxplot(x='cluster', y='mag', data=df_clean, palette='Set2', ax=axes_box[0])
        axes_box[0].set_title('Magnitudo')
        sns.boxplot(x='cluster', y='depth', data=df_clean, palette='Set2', ax=axes_box[1])
        axes_box[1].set_title('Kedalaman')
        plt.tight_layout()
        st.pyplot(fig_box)

    with col6:
        st.subheader("6. 3D Scatter Plot (Lon, Lat, Depth)")
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        scatter = ax_3d.scatter(df_clean['longitude'], df_clean['latitude'], df_clean['depth'],
                                c=df_clean['cluster'], cmap='viridis', s=10, alpha=0.6)
        ax_3d.set_xlabel('Longitude')
        ax_3d.set_ylabel('Latitude')
        ax_3d.set_zlabel('Depth')
        ax_3d.set_title('3D Plot Persebaran Gempa')
        st.pyplot(fig_3d)

    # Baris 4: Boxplot Terpisah Magnitudo & Kedalaman
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("7. Boxplot Magnitudo per Cluster")
        fig_bm, ax_bm = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_clean, x='cluster', y='mag', palette='viridis', ax=ax_bm)
        ax_bm.set_title('Earthquake Magnitude Distribution by Cluster')
        st.pyplot(fig_bm)

    with col8:
        st.subheader("8. Boxplot Kedalaman per Cluster")
        fig_bd, ax_bd = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_clean, x='cluster', y='depth', palette='viridis', ax=ax_bd)
        ax_bd.set_title('Earthquake Depth Distribution by Cluster')
        st.pyplot(fig_bd)

    # 6. Visualisasi Metode Evaluasi (Elbow & Silhouette Range)
    st.markdown("---")
    st.header("📈 Analisis Penentuan K Optimal (Elbow & Silhouette Method)")
    
    col_e1, col_e2 = st.columns(2)
    K_range = range(2, 8)

    with col_e1:
        st.subheader("9. Elbow Method (Inertia)")
        inertia = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        
        fig_elb, ax_elb = plt.subplots(figsize=(8, 5))
        ax_elb.plot(list(K_range), inertia, marker='o', color='blue', linestyle='-')
        ax_elb.set_title('Elbow Method')
        ax_elb.set_xlabel('Number of clusters (K)')
        ax_elb.set_ylabel('Inertia')
        st.pyplot(fig_elb)

    with col_e2:
        st.subheader("10. Silhouette Score Method")
        sil_scores = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))
        
        fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
        ax_sil.plot(list(K_range), sil_scores, marker='s', color='green', linestyle='-')
        ax_sil.set_title('Silhouette Score Method')
        ax_sil.set_xlabel('Number of clusters (K)')
        ax_sil.set_ylabel('Silhouette Score')
        ax_sil.grid(True)
        st.pyplot(fig_sil)

except FileNotFoundError:
    st.error("File 'global_earthquakes_seismic.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan skrip ini.")