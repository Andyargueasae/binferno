import numpy as np
import hdbscan
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import KernelPCA
from umap import UMAP

# 1. 定义 L2 归一化函数（对比学习空间必备）
def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)

# 2. 构建针对对比学习隐空间的先进 Pipeline
simclr_hdbscan_pipeline = Pipeline([
    # 第一步：确保所有隐向量严格位于超球面上
    ('l2_norm', FunctionTransformer(l2_normalize)),
    
    # 第二步：使用基于余弦相似度的核 PCA 降维到 50 维
    # 相比普通 PCA，这能完美保留超球面上的夹角结构并过滤隐空间噪声
    ('cosine_pca', KernelPCA(n_components=50, kernel='cosine', random_state=42)),
    
    # 第三步：用 UMAP 压榨到低维空间（5维是 HDBSCAN 的黄金伴侣）
    ('umap', UMAP(
        n_components=5,          
        n_neighbors=30,          # 略微调大（如 30-50），以更好地保持超球面的全局连续性
        min_dist=0.0,            # 严格设为 0.0！这能迫使同一流形的数据极度紧密地凝聚在一起
        metric='cosine',         # 核心：必须使用余弦距离度量
        random_state=42,
        n_jobs=-1
    ))
])

# 3. 执行降维流形嵌入
# X_simclr_latents 形状: (n_samples, 128)
low_dim_embeddings = simclr_hdbscan_pipeline.fit_transform(X_simclr_latents)

# 4. 配置并运行 HDBSCAN 聚类
# 推荐显式使用 'borrowed_target_' 策略或标准 euclidean 
# 因为 UMAP 映射后的低维空间已经转为了欧氏局部空间
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,         # 最小簇大小，根据你的数据集总体量调整
    min_samples=5,               # 控制保守度，值越小，被划分为噪声（-1）的样本越少
    metric='euclidean',          # UMAP 输出后，直接使用欧氏距离即可
    cluster_selection_method='eom' # 'eom' (Excess of Mass) 适合寻找紧凑集群
)

labels = clusterer.fit_predict(low_dim_embeddings)
