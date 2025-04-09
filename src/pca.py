import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation
    
    PCA is a dimensionality reduction technique that finds the directions (principal components)
    that maximize the variance in the data.
    
    Parameters:
    -----------
    n_components : int, optional (default=None)
        Number of components to keep. If None, all components are kept.
        
    Attributes:
    -----------
    components_ : ndarray of shape (n_components, n_features)
        Principal components (eigenvectors of the covariance matrix)
        
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component (eigenvalues of the covariance matrix)
        
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component
        
    mean_ : ndarray of shape (n_features,)
        Mean of the training data
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X):
        """
        Fit the model with X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Calculate the mean for each feature 
        self.mean_ = np.mean(X, axis=0).reshape(1, -1)
        
        # Shift the points to the new center
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        if self.n_components is None:
            self.n_components = X.shape[1]
        
        # Store the principal components
        self.components_ = Vt[:self.n_components]

        full_variances = S ** 2 / (X.shape[0] - 1)

        val = np.sum(full_variances)

        self.explained_variance_ = full_variances[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(full_variances)
        
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.
        
        Parameters:
        -----------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space
        """
        return np.dot(X_transformed, self.components_) + self.mean_