import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def main():
    print("Memulai analisis clustering...")
    
    # 1. Load Data
    print("Memuat dataset...")
    try:
        df = pd.read_csv('global_earthquakes_seismic.csv')
    except FileNotFoundError:
        print("File 'global_earthquakes_seismic.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return
        
    num_cols = ['latitude', 'longitude', 'depth', 'mag', 'sig', 'rms', 'gap']
    df_num = df[num_cols].dropna().drop_duplicates()
    
    # 2. Preprocessing (Membersihkan Outlier dengan IQR)
    print("Membersihkan data (outliers)...")
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_num[~((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
    
    # 3. Scaling
    print("Standarisasi data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # 4. K-Means Clustering (K=3)
    print("Menjalankan K-Means Clustering (K=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Menampilkan metrik
    sil_score = silhouette_score(X_scaled, df_clean['cluster'])
    print(f"\nSilhouette Score (K=3): {sil_score:.4f}")
    
    # 5. Visualisasi Lengkap (Menyesuaikan dengan file ipynb)
    print("\nMenghasilkan dan menyimpan visualisasi...")
    
    # a. Correlation Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df_clean.drop('cluster', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # b. Elbow Method
    inertia = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertia.append(kmeans_temp.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), inertia, marker='o', color='blue', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig('elbow_method.png')
    plt.close()
    
    # c. Silhouette Score Method
    silhouette_scores = []
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
    
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), silhouette_scores, marker='s', color='green', linestyle='-')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('silhouette_score_method.png')
    plt.close()
    
    # d. 2D Scatter Plot (Longitude vs Latitude)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df_clean, palette='viridis', s=15, alpha=0.8)
    plt.title('2D Scatter Plot of Clusters (Longitude vs Latitude)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster')
    plt.savefig('scatter_2d_lat_lon.png')
    plt.close()
    
    # e. PCA 2D Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['pca_1', 'pca_2'])
    df_pca['cluster'] = df_clean['cluster'].values
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='pca_1', y='pca_2', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('K-Means Clustering (PCA 2D Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('pca_2d_projection.png')
    plt.close()
    
    # f. Geospatial Distribution Scatter Plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_clean, x='longitude', y='latitude', hue='cluster', palette='Set1', s=15, alpha=0.7)
    plt.title('Geospatial Distribution of Earthquake Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('geospatial_distribution.png')
    plt.close()
    
    # g. Boxplots (Magnitudo & Depth per Cluster)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x='cluster', y='mag', data=df_clean, palette='Set2', ax=axes[0])
    axes[0].set_title('Distribusi Magnitudo (mag) per Cluster')
    sns.boxplot(x='cluster', y='depth', data=df_clean, palette='Set2', ax=axes[1])
    axes[1].set_title('Distribusi Kedalaman (depth) per Cluster')
    plt.tight_layout()
    plt.savefig('boxplots_mag_depth.png')
    plt.close()
    
    # Separate individual boxplots as in ipynb
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='cluster', y='mag', palette='viridis')
    plt.title('Earthquake Magnitude Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('boxplot_magnitude_cluster.png')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='cluster', y='depth', palette='viridis')
    plt.title('Earthquake Depth Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('boxplot_depth_cluster.png')
    plt.close()
    
    # h. 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_clean['longitude'], df_clean['latitude'], df_clean['depth'],
                         c=df_clean['cluster'], cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth')
    ax.set_title('3D Plot Persebaran Gempa (Lon, Lat, Depth)')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('scatter_3d_output.png')
    plt.close()
    
    print("Analisis dan seluruh visualisasi dari Jupyter Notebook telah berhasil ditambahkan ke file .py!")

if __name__ == '__main__':
    main()