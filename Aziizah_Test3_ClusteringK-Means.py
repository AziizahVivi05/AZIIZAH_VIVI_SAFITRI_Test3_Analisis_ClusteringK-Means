import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    
    # 4. K-Means Clustering
    print("Menjalankan K-Means Clustering (K=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Menampilkan metrik
    sil_score = silhouette_score(X_scaled, df_clean['cluster'])
    print(f"\nSilhouette Score: {sil_score:.4f}")
    print("Rata-rata karakteristik per cluster:")
    print(df_clean.groupby('cluster').mean())
    
    # 5. Visualisasi
    print("\nMenghasilkan visualisasi...")
    
    # a. Correlation Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df_clean.drop('cluster', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('heatmap_output.png')
    plt.close()
    
    # b. 2D Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df_clean, palette='viridis', s=15, alpha=0.8)
    plt.title('2D Scatter Plot of Clusters (Longitude vs Latitude)')
    plt.savefig('scatter_2d_output.png')
    plt.close()
    
    # c. Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x='cluster', y='mag', data=df_clean, palette='Set2', ax=axes[0])
    axes[0].set_title('Distribusi Magnitudo (mag) per Cluster')
    sns.boxplot(x='cluster', y='depth', data=df_clean, palette='Set2', ax=axes[1])
    axes[1].set_title('Distribusi Kedalaman (depth) per Cluster')
    plt.tight_layout()
    plt.savefig('boxplots_output.png')
    plt.close()
    
    # d. 3D Scatter Plot
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
    
    print("Analisis selesai. Visualisasi telah disimpan sebagai file .png di direktori yang sama.")

if __name__ == '__main__':
    main()