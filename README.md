# California Housing Unsupervised Machine Learning Analysis

> **Neighborhood Market Segmentation, Dimensionality Reduction, and Spatial Anomaly Detection using PCA, K-Means, and DBSCAN.**

---

## 📌 Project Overview

This project explores the **California Housing Dataset** from an **Unsupervised Machine Learning** perspective. Rather than building a traditional supervised regression model to predict prices, this project focuses on discovering hidden patterns, demographic segments, and geographic realities across California's 20,640 census blocks.

By combining **Principal Component Analysis (PCA)**, **K-Means Clustering**, and **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, this project accomplishes three primary objectives:
1. **Demographic & Market Segmentation:** Grouping California neighborhoods into actionable socioeconomic "personas."
2. **Dimensionality Reduction:** Compressing a 17-dimensional feature space into interpretable 2D components.
3. **Spatial & Data Anomaly Detection:** Isolating extreme real estate anomalies (resorts, group quarters, institutional housing, and recording errors).

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Environment:** Jupyter Notebook
- **Core Libraries:**
  - `pandas`, `numpy` — Data manipulation & ratio engineering
  - `matplotlib`, `seaborn` — Geospatial and mathematical visualizations
  - `scikit-learn` — Preprocessing (`StandardScaler`, `OneHotEncoder`), Dimensionality Reduction (`PCA`), Clustering (`KMeans`, `DBSCAN`), Evaluation (`silhouette_score`, `NearestNeighbors`)

---

## 🚀 End-to-End Workflow

### 1. Data Cleaning & Foundation
- Checked data distribution and identified missing values in `total_bedrooms`.
- Performed robust median imputation (`435.0` bedrooms) to prevent distribution distortion.

### 2. Feature Engineering & Preprocessing
- **Engineered Real Estate Ratios:**
  - `rooms_per_household = total_rooms / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms`
  - `population_per_household = population / households`
- **Categorical Encoding:** One-Hot Encoded `ocean_proximity` (5 categorical dummies), expanding the feature space to 17 numeric columns.
- **Feature Scaling:** Applied `StandardScaler` to ensure fair distance measurements across all features for distance-based clustering.

### 3. Dimensionality Reduction (PCA)
- Computed 17 Principal Components and evaluated the explained variance scree plot.
- Projected the dataset into a 2D space (`PC1` vs `PC2`), revealing that `PC1` captures overall size/volume features (`total_rooms`, `households`, `population`), while `PC2` captures density and ratio dynamics.

### 4. Partition Clustering: K-Means ($K=5$)
- Evaluated $K \in [2, 8]$ using **Inertia (Elbow Method)** and **Silhouette Score**.
- Identified $K=5$ as the mathematically superior and domain-sound choice, reflecting the natural economic quadrants of California.
- Profiled the 5 neighborhood personas:
  - **Cluster 1 (Inland Working Class):** Lowest average house value (~$121,000) and income (~$31,300), larger homes (5.95 rooms), and highest family household size (3.33).
  - **Cluster 3 (Historic Bay Area Core):** Highest house value (~$259,000), oldest housing stock (38.3 years), and smallest household sizes (2.62 people).
  - **Cluster 4 (Master-Planned Suburbs):** Highest median income (~$42,700), youngest developments (15.4 years), and larger homes.
  - **Cluster 0 (Established SoCal Metros):** High-density coastal living across Los Angeles and San Diego (~$240,000 average value).
  - **Cluster 2 (Pacific Coastal Communities):** Scenic coastal pockets along the Pacific Highway (~$248,000 average value).

### 5. Density-Based Clustering & Outlier Detection: DBSCAN
- Built a **K-Distance Graph** (10th nearest neighbor) to locate the inflection point ("knee") at $\epsilon = 2.5$.
- Fit DBSCAN (`eps=2.5`, `min_samples=10`), finding 5 natural core clusters and isolating **196 extreme outliers (`-1` / 0.95% of data)**.
- **Outlier Interrogation:** Revealed that the isolated noise points were not standard family neighborhoods, but rather:
  - Mega-resorts and hotels (up to **141.9 rooms per household**).
  - Group quarters, college dorms, or barracks (up to **1,243 people per household**).
  - Census blocks with populations reaching **35,682 people**.

---

## 📊 Model Comparison: K-Means vs. DBSCAN

| Feature / Metric | K-Means | DBSCAN |
| :--- | :--- | :--- |
| **Model Type** | Partition / Centroid-Based | Density-Based Spatial Clustering |
| **Clusters Found** | 5 (Chosen via Elbow & Silhouette peak) | 5 Dense Clusters + 196 Outliers |
| **Handling of Outliers** | Forces outliers into nearest centroid (Vulnerable) | Explicitly flags noise as `-1` (Very Robust) |
| **Cluster Geometry** | Assumes spherical / round shapes | Discovers arbitrary, natural shapes (follows coasts/valleys) |
| **Best Real Estate Use Case** | Broad market segmentation & buyer personas | Anomaly detection, resort identification & data cleaning |

---

## 📁 Repository Structure

```text
├── housing.csv                     # Original California Housing dataset
├── housing_prediction.ipynb        # Complete Jupyter notebook with code & visualizations
├── housing_clustered_results.csv   # Final dataset enriched with K-Means & DBSCAN cluster labels
└── README.md                       # Project documentation & summary
```

---

## 👤 Author & Contact

**Project Lead:** Melchizedek Madaki  
**Email:** [melchizedekmadaki93@gmail.com](mailto:melchizedekmadaki93@gmail.com)  

*Feel free to reach out for questions, collaborations, or discussions on Data Science and Machine Learning!*
