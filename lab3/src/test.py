import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from safetensors.numpy import save_file, load_file
from pathlib import Path
from typing import Union
from PIL import Image
import json

class PCA:
    """
    Principal Component Analysis.

    We take X as the input data, which is a matrix of shape (N, D), where N is the number of samples and D is the dimension of the data.

    Args:
        - dim: int, Number of components to keep. Marked as d.

    Parameters:
        - components: np.ndarray, shape (d, D), Principal components.
        - mean: np.ndarray, shape (D,), Mean of the data.

    Methods:
        - fit(X): Fit the PCA model to the data.
        - transform(X): Project the data into the reduced space.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.components = None
        self.mean = None

    @classmethod
    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the data.

        Args:
            - X: np.ndarray, shape (N, D), Data.
        """
        # 2.1 Compute the principal components
        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project the data into the reduced space.

        Args:
            - X: np.ndarray, shape (N, D), Data.

        Returns:
            - X_pca: np.ndarray, shape (N, d), Projected data.
        """
        # Project data
        X = X - self.mean
        return X @ self.components.T

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Project the data back to the original space.

        Args:
            - X_pca: np.ndarray, shape (N, d), Projected data.

        Returns:
            - X: np.ndarray, shape (N, D), Original data.
        """
        return X_pca @ self.components + self.mean


class PCA:
    """
    Principal Component Analysis.

    We take X as the input data, which is a matrix of shape (N, D), where N is the number of samples and D is the dimension of the data.

    Args:
        - dim: int, Number of components to keep. Marked as d.

    Parameters:
        - components: np.ndarray, shape (d, D), Principal components.
        - mean: np.ndarray, shape (D,), Mean of the data.

    Methods:
        - fit(X): Fit the PCA model to the data.
        - transform(X): Project the data into the reduced space.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the data.

        Args:
            - X: np.ndarray, shape (N, D), Data.
        """
        # 2.1 Compute the principal components
        N, D = X.shape

        # Compute the mean of the data
        self.mean = np.mean(X, axis=0)  # [D,]

        # Center the data
        X_centered = X - self.mean  # [N, D]

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)  # [D, D]

        # Perform eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # eigvals: [D,], eigvecs: [D, D]

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Select the top `dim` principal components
        self.components = eigvecs[:, :self.dim].T  # [d, D]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project the data into the reduced space.

        Args:
            - X: np.ndarray, shape (N, D), Data.

        Returns:
            - X_pca: np.ndarray, shape (N, d), Projected data.
        """
        # Project data
        X = X - self.mean
        return X @ self.components.T

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Project the data back to the original space.

        Args:
            - X_pca: np.ndarray, shape (N, d), Projected data.

        Returns:
            - X: np.ndarray, shape (N, D), Original data.
        """
        return X_pca @ self.components + self.mean


def sample_from_gmm(gmm: GMM, pca: PCA, label: int, path: Union[str, Path]):
    """
    Sample images from a Gaussian Mixture Model.

    Args:
        - gmm: GMM, Gaussian Mixture Model.
        - pca: PCA, Principal Component Analysis.
        - label: int, Cluster label.
        - path: Union[str, Path], Path to save the sampled image.
    """
    # 从 GMM 的指定高斯分量中采样一个样本
    mean = gmm.means[label]  # 获取对应标签的均值 [D,]
    cov = gmm.covs[label]    # 获取对应标签的协方差矩阵 [D, D]
    sample = np.random.multivariate_normal(mean, cov)  # 从该分布中采样一个点 [D,]

    # 使用 PCA 的 inverse_transform 方法将样本还原到原始空间 [H * W,]
    sample_original = pca.inverse_transform(sample.reshape(1, -1))  # [1, H*W]

    # 将样本 reshape 成图片大小（假设 MNIST 数据集 H=W=28）
    sample_image = sample_original.reshape(28, 28)  # [H, W]

    # 归一化像素值到 [0, 255]
    sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min()) * 255
    sample_image = sample_image.astype(np.uint8)  # 转换为整数类型

    # 保存图片
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  # 确保路径存在
    sample = Image.fromarray(sample_image, mode="L")  # 创建灰度图
    sample.save(path / "gmm_sample.png")