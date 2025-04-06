import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.feature import hog, local_binary_pattern
import warnings
import time
import random
from matplotlib.colors import ListedColormap
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class GarbageClassifier:
    """
    A class for classifying garbage images using traditional machine learning techniques.
    """
    
    def __init__(self, base_path='reference_images', n_samples_per_class=500):
        """
        Initialize the GarbageClassifier.
        
        Parameters:
        -----------
        base_path : str
            Path to the directory containing garbage image categories
        n_samples_per_class : int
            Number of samples to use from each class for balanced dataset
        """
        self.base_path = base_path
        self.n_samples_per_class = n_samples_per_class
        self.classes = []
        self.images = []
        self.labels = []
        self.features = {}
        self.models = {}
        self.ensemble_model = None
        
    def load_dataset(self):
        """
        Load images from the directory structure and create a balanced dataset.
        Each subdirectory is treated as a separate class.
        
        Returns:
        --------
        tuple
            (images, labels, classes)
        """
        print("Loading dataset...")
        
        # Check if base path exists
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Base path {self.base_path} not found. Please ensure the directory exists.")
        
        # Get all subdirectories (classes)
        self.classes = [d for d in os.listdir(self.base_path) 
                        if os.path.isdir(os.path.join(self.base_path, d))]
        
        if not self.classes:
            raise ValueError(f"No subdirectories found in {self.base_path}. Please ensure your dataset is properly organized.")
        
        print(f"Found {len(self.classes)} garbage categories: {self.classes}")
        
        # Define a common size for all images to avoid array shape issues
        target_size = (128, 128)
        
        # Dictionary to store the original and sampled counts
        class_stats = {}
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.base_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"Warning: No images found in class '{class_name}'. Skipping this class.")
                class_stats[class_name] = {'original': 0, 'sampled': 0}
                continue
            
            # Store original count
            original_count = len(image_files)
            
            # Limit the number of samples per class for balance
            if len(image_files) > self.n_samples_per_class:
                image_files = random.sample(image_files, self.n_samples_per_class)
                sampled_count = self.n_samples_per_class
            else:
                sampled_count = original_count
            
            # Store the counts
            class_stats[class_name] = {'original': original_count, 'sampled': sampled_count}
            
            print(f"Loading {len(image_files)} images from class '{class_name}'")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                # Read image and convert to RGB
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Failed to load image {img_path}. Skipping this image.")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize to common dimensions immediately to ensure all images have the same shape
                    img = cv2.resize(img, target_size)
                    self.images.append(img)
                    self.labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}. Skipping this image.")
        
        if not self.images:
            raise ValueError("No valid images were loaded. Please check your dataset.")
            
        # Convert lists to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Dataset loaded: {len(self.images)} images, {len(self.classes)} classes")
        print(f"Images shape: {self.images.shape}")
        
        # Visualize the dataset before and after sampling
        self.visualize_dataset_sampling(class_stats, output_dir='output')
        
        return self.images, self.labels, self.classes
        
    def visualize_dataset_sampling(self, class_stats, output_dir='output'):
        """
        Visualize the dataset before and after random sampling.
        
        Parameters:
        -----------
        class_stats : dict
            Dictionary containing original and sampled counts for each class
        output_dir : str
            Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract data for plotting
        classes = list(class_stats.keys())
        original_counts = [stats['original'] for stats in class_stats.values()]
        sampled_counts = [stats['sampled'] for stats in class_stats.values()]
        
        # Calculate the difference
        differences = [orig - samp for orig, samp in zip(original_counts, sampled_counts)]
        
        # Create figure for comparison
        plt.figure(figsize=(14, 8))
        
        # Create a grouped bar chart
        x = np.arange(len(classes))
        width = 0.35
        
        # Plot bars
        plt.bar(x - width/2, original_counts, width, label='Original Dataset', color='skyblue')
        plt.bar(x + width/2, sampled_counts, width, label='Sampled Dataset', color='lightgreen')
        
        # Add labels, title and legend
        plt.xlabel('Garbage Category')
        plt.ylabel('Number of Images')
        plt.title('Dataset Before and After Random Sampling)')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        
        # Add count labels above the bars
        for i, (orig, samp) in enumerate(zip(original_counts, sampled_counts)):
            plt.text(i - width/2, orig + 5, str(orig), ha='center', fontweight='bold')
            plt.text(i + width/2, samp + 5, str(samp), ha='center', fontweight='bold')
            
            # Add difference label if there is any
            if orig != samp:
                plt.text(i, min(orig, samp) / 2, f"-{orig-samp}", ha='center', 
                        color='red', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'dataset_sampling_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset sampling comparison to {output_path}")
        plt.close()
        
        # Create a table visualization
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        
        # Prepare data for table
        table_data = []
        for class_name, stats in class_stats.items():
            orig = stats['original']
            samp = stats['sampled']
            diff = orig - samp
            percent = (diff / orig * 100) if orig > 0 else 0
            
            table_data.append([
                class_name, 
                str(orig), 
                str(samp), 
                f"{diff} ({percent:.1f}%)" if diff > 0 else "0 (0.0%)"
            ])
        
        # Create table
        table = plt.table(
            cellText=table_data,
            colLabels=['Category', 'Original Count', 'Sampled Count', 'Difference(%)'],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.2, 0.2, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Dataset Sampling Statistics', fontsize=16)
        
        # Save table
        table_output_path = os.path.join(output_dir, 'dataset_sampling_table.png')
        plt.savefig(table_output_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset sampling table to {table_output_path}")
        plt.close()
    
    def preprocess_images(self, images, target_size=(128, 128)):
        """
        Preprocess images by converting to grayscale for feature extraction.
        Since images are already resized during loading, we only need to convert to grayscale.
        
        Parameters:
        -----------
        images : ndarray
            Array of images to preprocess
        target_size : tuple
            Target size for resizing (used only if images aren't already at this size)
            
        Returns:
        --------
        ndarray
            Preprocessed images
        """
        preprocessed = []
        for img in images:
            # Check if resize is needed (in case images are already resized)
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size)
            
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            preprocessed.append(gray)
        
        return np.array(preprocessed)
    
    def extract_hog_features(self, images):
        """
        Extract Histogram of Oriented Gradients (HOG) features from images.
        
        Parameters:
        -----------
        images : ndarray
            Array of grayscale images
            
        Returns:
        --------
        ndarray
            HOG features for each image
        """
        print("Extracting HOG features...")
        features = []
        visualizations = []
        
        total_images = len(images)
        for i, img in enumerate(images):
            if i % 50 == 0 or i == total_images - 1:
                print(f"  Processing image {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)")
                
            hog_features, hog_image = hog(
                img, 
                orientations=8, 
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), 
                visualize=True, 
                block_norm='L2-Hys'
            )
            features.append(hog_features)
            visualizations.append(hog_image)
        
        print(f"HOG features shape: {np.array(features).shape}")
        self.features['hog'] = np.array(features)
        return np.array(features), np.array(visualizations)
    
    def extract_lbp_features(self, images, radius=3, n_points=24):
        """
        Extract Local Binary Pattern (LBP) features from images.
        
        Parameters:
        -----------
        images : ndarray
            Array of grayscale images
        radius : int
            Radius parameter for LBP
        n_points : int
            Number of points parameter for LBP
            
        Returns:
        --------
        ndarray
            LBP features for each image
        """
        print("Extracting LBP features...")
        features = []
        visualizations = []
        
        total_images = len(images)
        for i, img in enumerate(images):
            if i % 50 == 0 or i == total_images - 1:
                print(f"  Processing image {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)")
                
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            # Calculate LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
            features.append(hist)
            visualizations.append(lbp)
        
        print(f"LBP features shape: {np.array(features).shape}")
        self.features['lbp'] = np.array(features)
        return np.array(features), np.array(visualizations)
    
    def extract_sift_features(self, images, n_keypoints=100):
        """
        Extract Scale-Invariant Feature Transform (SIFT) features from images.
        Use Bag of Visual Words approach to create fixed-length feature vectors.
        
        Parameters:
        -----------
        images : ndarray
            Array of grayscale images
        n_keypoints : int
            Maximum number of keypoints to use
            
        Returns:
        --------
        ndarray
            SIFT bag of visual words features for each image
        """
        print("Extracting SIFT features...")
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Extract SIFT keypoints and descriptors from all images
        all_descriptors = []
        keypoint_images = []
        
        total_images = len(images)
        for i, img in enumerate(images):
            if i % 50 == 0 or i == total_images - 1:
                print(f"  Detecting keypoints for image {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)")
                
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            # Create visualization
            img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            keypoint_images.append(img_with_keypoints)
            
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # Combine all descriptors
        all_descriptors = np.vstack(all_descriptors)
        
        # Create bag of visual words using K-means clustering
        from sklearn.cluster import KMeans
        
        # Use a smaller sample of descriptors for clustering if there are too many
        max_desc_for_clustering = 10000
        if all_descriptors.shape[0] > max_desc_for_clustering:
            indices = np.random.choice(all_descriptors.shape[0], max_desc_for_clustering, replace=False)
            desc_sample = all_descriptors[indices]
        else:
            desc_sample = all_descriptors
        
        n_visual_words = 100
        print(f"Clustering {desc_sample.shape[0]} SIFT descriptors into {n_visual_words} visual words...")
        
        kmeans = KMeans(n_clusters=n_visual_words, random_state=42, n_init=10)
        kmeans.fit(desc_sample)
        
        # Create histogram of visual words for each image
        bow_features = []
        
        for img in images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is not None:
                # Assign each descriptor to a visual word
                visual_word_labels = kmeans.predict(descriptors)
                
                # Create histogram of visual words
                hist, _ = np.histogram(visual_word_labels, bins=n_visual_words, range=(0, n_visual_words - 1))
                
                # Normalize histogram
                if np.sum(hist) > 0:
                    hist = hist / np.sum(hist)
                
                bow_features.append(hist)
            else:
                # If no descriptors found, use zero histogram
                bow_features.append(np.zeros(n_visual_words))
        
        bow_features = np.array(bow_features)
        print(f"SIFT BoW features shape: {bow_features.shape}")
        self.features['sift'] = bow_features
        
        return bow_features, np.array(keypoint_images)
    
    def visualize_features(self, original_images, hog_images, lbp_images, sift_images, n_samples=3, output_dir='output'):
        """
        Visualize the original images and their extracted features.
        
        Parameters:
        -----------
        original_images : ndarray
            Original images
        hog_images : ndarray
            HOG feature visualizations
        lbp_images : ndarray
            LBP feature visualizations
        sift_images : ndarray
            SIFT keypoint visualizations
        n_samples : int
            Number of samples to visualize
        output_dir : str
            Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Randomly select samples
        indices = np.random.choice(len(original_images), n_samples, replace=False)
        
        plt.figure(figsize=(15, 4 * n_samples))
        
        for i, idx in enumerate(indices):
            # Original image
            plt.subplot(n_samples, 4, i*4 + 1)
            plt.imshow(original_images[idx])
            plt.title('Original Image')
            plt.axis('off')
            
            # HOG features
            plt.subplot(n_samples, 4, i*4 + 2)
            plt.imshow(hog_images[idx], cmap='gray')
            plt.title('HOG Features')
            plt.axis('off')
            
            # LBP features
            plt.subplot(n_samples, 4, i*4 + 3)
            plt.imshow(lbp_images[idx], cmap='gray')
            plt.title('LBP Features')
            plt.axis('off')
            
            # SIFT features
            plt.subplot(n_samples, 4, i*4 + 4)
            plt.imshow(sift_images[idx])
            plt.title('SIFT Features')
            plt.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'feature_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature visualization to {output_path}")
        plt.close()
    
    def visualize_feature_space(self, features, labels, classes, feature_name, n_components=2, output_dir='output'):
        """
        Visualize the feature space after dimensionality reduction.
        
        Parameters:
        -----------
        features : ndarray
            Feature vectors
        labels : ndarray
            Corresponding labels
        classes : list
            Class names
        feature_name : str
            Name of the feature extraction method
        n_components : int
            Number of components for dimensionality reduction
        output_dir : str
            Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use t-SNE for dimensionality reduction
        print(f"Reducing {feature_name} features to {n_components}D using t-SNE...")
        
        # Apply PCA first to reduce dimensionality before t-SNE
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, features.shape[1]), random_state=42)
            reduced_features = pca.fit_transform(features)
        else:
            reduced_features = features
        
        tsne = TSNE(n_components=n_components, random_state=42, learning_rate='auto', init='pca')
        reduced_features = tsne.fit_transform(reduced_features)
        
        # Create color map
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Plot the feature space
        plt.figure(figsize=(10, 8))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                       color=colors[i], label=classes[label], alpha=0.7)
        
        plt.title(f'Images Under the {feature_name.upper()} Feature Space')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_path = os.path.join(output_dir, f'{feature_name}_feature_space.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {feature_name} feature space visualization to {output_path}")
        plt.close()
    
    def train_model(self, X_train, y_train, model_type='svm', feature_type=None, cv=5):
        """
        Train a classifier model with cross-validation.
        
        Parameters:
        -----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training labels
        model_type : str
            Type of model to train ('svm', 'knn', 'rf')
        feature_type : str
            Type of features used (for model naming)
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        object
            Trained model
        """
        model_name = f"{feature_type}_{model_type}" if feature_type else model_type
        print(f"Training {model_name} model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Select model based on type
        if model_type == 'svm':
            # Define parameter grid for grid search
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['linear', 'rbf']
            }
            model = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=cv, n_jobs=-1)
            
        elif model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1 for manhattan, 2 for euclidean
            }
            model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, n_jobs=-1)
            
        elif model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, n_jobs=-1)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model with grid search
        model.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {model.best_params_}")
        print(f"Cross-validation accuracy: {model.best_score_:.4f}")
        
        # Store model and scaler
        self.models[model_name] = {
            'model': model.best_estimator_,
            'scaler': scaler,
            'accuracy': model.best_score_
        }
        
        return model.best_estimator_, scaler
    
    def create_ensemble(self, X_train, y_train):
        """
        Create an ensemble model from the trained models.
        
        Parameters:
        -----------
        X_train : ndarray
            Combined training features
        y_train : ndarray
            Training labels
            
        Returns:
        --------
        object
            Trained ensemble model
        """
        print("Creating ensemble model...")
        
        # Create a list of (name, model) tuples for VotingClassifier
        estimators = []
        
        for model_name, model_info in self.models.items():
            estimators.append((model_name, model_info['model']))
        
        # Create and train voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        self.ensemble_model = ensemble
        return ensemble
    
    def evaluate_model(self, model, scaler, X_test, y_test, class_names, model_name=None, output_dir='output'):
        """
        Evaluate a model's performance on test data.
        
        Parameters:
        -----------
        model : object
            Trained model
        scaler : object
            Fitted scaler
        X_test : ndarray
            Test features
        y_test : ndarray
            Test labels
        class_names : list
            Names of the classes
        model_name : str
            Name of the model for display purposes
        output_dir : str
            Directory to save evaluation results
            
        Returns:
        --------
        float
            Accuracy of the model on test data
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Scale test data if scaler is provided
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel: {model_name}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        # Save classification report to file
        report_path = os.path.join(output_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Test accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Saved classification report to {report_path}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {cm_path}")
        plt.close()
        
        return accuracy
    
    def generate_performance_table(self, output_dir='output'):
        """
        Generate a table of model performance metrics.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the performance table
            
        Returns:
        --------
        pd.DataFrame
            Table of model performance metrics
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        models_data = []
        
        for model_name, model_info in self.models.items():
            models_data.append({
                'Model/Classifier': model_name,
                'Accuracy': f"{model_info['accuracy'] * 100:.2f}%"
            })
        
        # Add ensemble model if available
        if hasattr(self, 'ensemble_accuracy'):
            models_data.append({
                'Model/Classifier': 'Ensemble Model',
                'Accuracy': f"{self.ensemble_accuracy * 100:.2f}%"
            })
        
        df = pd.DataFrame(models_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'model_performance.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved performance metrics to {csv_path}")
        
        # Plot as a table
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        table = plt.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.5, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        plt.title('Model Performance Comparison', fontsize=16)
        
        table_path = os.path.join(output_dir, 'model_performance_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance table visualization to {table_path}")
        plt.close()
        
        return df

    def test_on_complex_scenes(self, complex_images_path, models_to_test, output_dir='output'):
        """
        Test multiple models on images with complex or cluttered scenes.
        Visualizes different feature extractions and shows predictions for all models.
        
        Parameters:
        -----------
        complex_images_path : str
            Path to directory containing complex scene images
        models_to_test : dict
            Dictionary of models to test, with keys as model names and values as (model, scaler) tuples
        output_dir : str
            Directory to save visualization results
            
        Returns:
        --------
        dict
            Predictions for each image from each model
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load and preprocess complex scene images
        complex_images = []
        image_files = []
        
        # Use the same target size as in load_dataset
        target_size = (128, 128)
        
        print(f"Loading complex scene images from {complex_images_path}...")
        
        for img_file in os.listdir(complex_images_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(complex_images_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize immediately to ensure consistent dimensions
                    img = cv2.resize(img, target_size)
                    complex_images.append(img)
                    image_files.append(img_file)
        
        if not complex_images:
            print("No complex scene images found")
            return {}
        
        print(f"Loaded {len(complex_images)} complex scene images")
        
        # Preprocess images
        preprocessed = self.preprocess_images(complex_images)
        
        # Extract features (all three feature types)
        print("Extracting features from complex scene images...")
        hog_features, hog_visualizations = self.extract_hog_features(preprocessed)
        lbp_features, lbp_visualizations = self.extract_lbp_features(preprocessed)
        sift_features, sift_visualizations = self.extract_sift_features(preprocessed)
        
        # Extract edges for visualization (Canny edge detector)
        edge_visualizations = []
        for img in preprocessed:
            edges = cv2.Canny(img, 100, 200)
            edge_visualizations.append(edges)
        
        # Dictionary to store results for each model
        all_results = {}
        
        # Test each model
        for model_name, (model, scaler) in models_to_test.items():
            print(f"\nTesting model: {model_name}")
            
            if model_name == 'ensemble':
                # For ensemble model, combine all feature types just like in training
                hog_scaled = models_to_test['hog_svm'][1].transform(hog_features)
                lbp_scaled = models_to_test['lbp_svm'][1].transform(lbp_features)
                sift_scaled = models_to_test['sift_svm'][1].transform(sift_features)
                
                # Combine all scaled features
                X_scaled = np.hstack((hog_scaled, lbp_scaled, sift_scaled))
                print(f"Using combined features (HOG+LBP+SIFT) for ensemble model. Features shape: {X_scaled.shape}")
            else:
                # For non-ensemble models, use the appropriate feature type
                if 'hog' in model_name:
                    features = hog_features
                elif 'lbp' in model_name:
                    features = lbp_features
                elif 'sift' in model_name:
                    features = sift_features
                else:
                    # Use HOG features as default for any other model
                    features = hog_features
                    print(f"Using HOG features for model {model_name}")
                
                # Scale features using the model's scaler
                if scaler is not None:
                    X_scaled = scaler.transform(features)
                else:
                    X_scaled = features
            
            # Make predictions with probabilities
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            # Get top 3 predictions for each image
            top_predictions = []
            for probs in probabilities:
                # Get indices of top 3 probabilities
                top_indices = probs.argsort()[-3:][::-1]
                # Get class names and probabilities for top 3
                top_3 = [(self.classes[idx], probs[idx] * 100) for idx in top_indices]
                top_predictions.append(top_3)
            
            # Store results
            all_results[model_name] = [(self.classes[pred], top_3) for pred, top_3 in zip(predictions, top_predictions)]
            
            # Create visualizations for this model
            for i, (img, pred, top_3, filename) in enumerate(
                    zip(complex_images, predictions, top_predictions, image_files)):
                
                # Create figure with 6 subplots (original, 4 feature visualizations, predictions)
                plt.figure(figsize=(18, 12))
                
                # Original image
                plt.subplot(2, 3, 1)
                plt.imshow(img)
                plt.title(f"Original: {filename}", fontsize=10)
                plt.axis('off')
                
                # HOG features
                plt.subplot(2, 3, 2)
                plt.imshow(hog_visualizations[i], cmap='gray')
                plt.title('HOG Features', fontsize=10)
                plt.axis('off')
                
                # LBP features
                plt.subplot(2, 3, 3)
                plt.imshow(lbp_visualizations[i], cmap='gray')
                plt.title('LBP Features', fontsize=10)
                plt.axis('off')
                
                # SIFT features
                plt.subplot(2, 3, 4)
                plt.imshow(sift_visualizations[i])
                plt.title('SIFT Features', fontsize=10)
                plt.axis('off')
                
                # Edge features
                plt.subplot(2, 3, 5)
                plt.imshow(edge_visualizations[i], cmap='gray')
                plt.title('Image Edges', fontsize=10)
                plt.axis('off')
                
                # Create a text subplot for predictions
                plt.subplot(2, 3, 6)
                plt.axis('off')
                plt.text(0.1, 0.8, f"Model: {model_name}", fontsize=12, fontweight='bold')
                plt.text(0.1, 0.7, f"Predicted: {self.classes[pred]}", fontsize=12, color='red')
                plt.text(0.1, 0.6, f"Top predictions:", fontsize=12, fontweight='bold')
                
                for j, (class_name, prob) in enumerate(top_3):
                    plt.text(0.1, 0.5 - j*0.1, f"{j+1}. {class_name}: {prob:.2f}%", fontsize=12)
                
                plt.tight_layout()
                
                # Save to output directory
                output_path = os.path.join(output_dir, f'complex_scene_{i+1}_{model_name}_analysis.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # Create a summary comparison across all models
        self._create_model_comparison_summary(complex_images, image_files, all_results, output_dir)
        
        return all_results

    def _create_model_comparison_summary(self, images, image_files, all_results, output_dir):
        """
        Create a summary visualization comparing predictions from all models.
        
        Parameters:
        -----------
        images : list
            List of complex scene images
        image_files : list
            List of image filenames
        all_results : dict
            Dictionary mapping model names to prediction results
        output_dir : str
            Directory to save the summary visualization
        """
        model_names = list(all_results.keys())
        
        for i, (img, filename) in enumerate(zip(images, image_files)):
            # Create a figure for this image comparing all models
            n_models = len(model_names)
            fig_height = 3 + n_models * 0.8
            plt.figure(figsize=(12, fig_height))
            
            # Display the original image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original: {filename}", fontsize=12)
            plt.axis('off')
            
            # Create a table of model predictions
            plt.subplot(1, 2, 2)
            plt.axis('off')
            
            plt.text(0.1, 0.95, "Model Predictions", fontsize=14, fontweight='bold')
            
            # Headers
            col_positions = [0.1, 0.5, 0.8]
            plt.text(col_positions[0], 0.9, "Model", fontsize=12, fontweight='bold')
            plt.text(col_positions[1], 0.9, "Prediction", fontsize=12, fontweight='bold')
            plt.text(col_positions[2], 0.9, "Confidence", fontsize=12, fontweight='bold')
            
            # Model predictions
            for j, model_name in enumerate(model_names):
                prediction, top_3 = all_results[model_name][i]
                confidence = top_3[0][1]  # Confidence of top prediction
                
                row_y = 0.85 - j * 0.05
                plt.text(col_positions[0], row_y, model_name, fontsize=10)
                plt.text(col_positions[1], row_y, prediction, fontsize=10)
                plt.text(col_positions[2], row_y, f"{confidence:.2f}%", fontsize=10)
            
            plt.tight_layout()
            
            # Save to output directory
            output_path = os.path.join(output_dir, f'complex_scene_{i+1}_model_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create an overall accuracy summary across all images
        self._create_model_accuracy_summary(image_files, all_results, output_dir)
    
    def _create_model_accuracy_summary(self, image_files, all_results, output_dir):
        """
        Create a summary of model predictions across all test images.
        
        Parameters:
        -----------
        image_files : list
            List of image filenames
        all_results : dict
            Dictionary mapping model names to prediction results
        output_dir : str
            Directory to save the summary visualization
        """
        model_names = list(all_results.keys())
        
        # Create a DataFrame to summarize predictions
        summary_data = []
        
        for i, filename in enumerate(image_files):
            row_data = {'Image': filename}
            
            for model_name in model_names:
                prediction, _ = all_results[model_name][i]
                row_data[f"{model_name}"] = prediction
                
            summary_data.append(row_data)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'complex_scenes_predictions.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved prediction summary to {csv_path}")
        
        # Create a visualization of the summary table
        plt.figure(figsize=(12, len(image_files)*0.5 + 2))
        plt.axis('off')
        
        # Create table
        table = plt.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.3] + [0.7/len(model_names)]*len(model_names)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Model Predictions on Complex Scenes', fontsize=16)
        
        # Save table
        table_path = os.path.join(output_dir, 'complex_scenes_summary_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction summary visualization to {table_path}")


def main():
    """
    Main function to run the garbage classification pipeline.
    """
    # Create output directory for saving figures
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create classifier instance
    classifier = GarbageClassifier(base_path='reference_images', n_samples_per_class=200)
    
    # Load dataset & Visualize class distribution
    images, labels, classes = classifier.load_dataset()

    # Preprocess images
    preprocessed_images = classifier.preprocess_images(images)
    
    # Extract features
    hog_features, hog_visualizations = classifier.extract_hog_features(preprocessed_images)
    lbp_features, lbp_visualizations = classifier.extract_lbp_features(preprocessed_images)
    sift_features, sift_visualizations = classifier.extract_sift_features(preprocessed_images)
    
    # Visualize features
    classifier.visualize_features(images, hog_visualizations, lbp_visualizations, sift_visualizations, output_dir=output_dir)
    
    # Visualize feature spaces
    classifier.visualize_feature_space(hog_features, labels, classes, 'hog', output_dir=output_dir)
    classifier.visualize_feature_space(lbp_features, labels, classes, 'lbp', output_dir=output_dir)
    classifier.visualize_feature_space(sift_features, labels, classes, 'sift', output_dir=output_dir)
    
    # Split data into train and test sets
    X_train_hog, X_test_hog, y_train, y_test = train_test_split(
        hog_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_lbp, X_test_lbp, _, _ = train_test_split(
        lbp_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_sift, X_test_sift, _, _ = train_test_split(
        sift_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train models on different features
    print("\n--- Training models ---")
    hog_svm, hog_svm_scaler = classifier.train_model(X_train_hog, y_train, 'svm', 'hog')
    hog_knn, hog_knn_scaler = classifier.train_model(X_train_hog, y_train, 'knn', 'hog')
    hog_rf, hog_rf_scaler = classifier.train_model(X_train_hog, y_train, 'rf', 'hog')
    
    lbp_svm, lbp_svm_scaler = classifier.train_model(X_train_lbp, y_train, 'svm', 'lbp')
    lbp_knn, lbp_knn_scaler = classifier.train_model(X_train_lbp, y_train, 'knn', 'lbp')
    lbp_rf, lbp_rf_scaler = classifier.train_model(X_train_lbp, y_train, 'rf', 'lbp')
    
    sift_svm, sift_svm_scaler = classifier.train_model(X_train_sift, y_train, 'svm', 'sift')
    sift_knn, sift_knn_scaler = classifier.train_model(X_train_sift, y_train, 'knn', 'sift')
    sift_rf, sift_rf_scaler = classifier.train_model(X_train_sift, y_train, 'rf', 'sift')
    
    # Evaluate models
    print("\n--- Evaluating models ---")
    classifier.evaluate_model(hog_svm, hog_svm_scaler, X_test_hog, y_test, classes, 'HOG+SVM', output_dir=output_dir)
    classifier.evaluate_model(hog_knn, hog_knn_scaler, X_test_hog, y_test, classes, 'HOG+KNN', output_dir=output_dir)
    classifier.evaluate_model(hog_rf, hog_rf_scaler, X_test_hog, y_test, classes, 'HOG+RF', output_dir=output_dir)

    classifier.evaluate_model(lbp_svm, lbp_svm_scaler, X_test_lbp, y_test, classes, 'LBP+SVM', output_dir=output_dir)
    classifier.evaluate_model(lbp_knn, lbp_knn_scaler, X_test_lbp, y_test, classes, 'LBP+KNN', output_dir=output_dir)
    classifier.evaluate_model(lbp_rf, lbp_rf_scaler, X_test_lbp, y_test, classes, 'LBP+RF', output_dir=output_dir)
    
    classifier.evaluate_model(sift_svm, sift_svm_scaler, X_test_sift, y_test, classes, 'SIFT+SVM', output_dir=output_dir)
    classifier.evaluate_model(sift_knn, sift_knn_scaler, X_test_sift, y_test, classes, 'SIFT+KNN', output_dir=output_dir)
    classifier.evaluate_model(sift_rf, sift_rf_scaler, X_test_sift, y_test, classes, 'SIFT+RF', output_dir=output_dir)
    
    # Create an ensemble model by combining features
    # Concatenate scaled features
    X_train_scaled_hog = hog_svm_scaler.transform(X_train_hog)
    X_train_scaled_lbp = lbp_rf_scaler.transform(X_train_lbp)
    X_train_scaled_sift = sift_svm_scaler.transform(X_train_sift)
    
    X_test_scaled_hog = hog_svm_scaler.transform(X_test_hog)
    X_test_scaled_lbp = lbp_rf_scaler.transform(X_test_lbp)
    X_test_scaled_sift = sift_svm_scaler.transform(X_test_sift)
    
    # Combine features into a single matrix
    X_train_combined = np.hstack((X_train_scaled_hog, X_train_scaled_lbp, X_train_scaled_sift))
    
    # Create ensemble classifier
    ensemble = classifier.create_ensemble(X_train_combined, y_train)
    
    # Evaluate ensemble on combined features
    X_test_combined = np.hstack((X_test_scaled_hog, X_test_scaled_lbp, X_test_scaled_sift))
    ensemble_accuracy = classifier.evaluate_model(
        ensemble, None, X_test_combined, y_test, classes, 'Ensemble', output_dir=output_dir
    )
    classifier.ensemble_accuracy = ensemble_accuracy
    
    # Generate performance table
    performance_table = classifier.generate_performance_table(output_dir=output_dir)
    
    # Test on complex scenes (if available)
    complex_scenes_path = 'complex_scenes'
    if os.path.exists(complex_scenes_path):
        print("\n--- Testing on complex scenes ---")
        
        # Create a dictionary of models to test
        models_to_test = {
            'hog_svm': (hog_svm, hog_svm_scaler),
            'hog_knn': (hog_knn, hog_knn_scaler),
            'hog_rf': (hog_rf, hog_rf_scaler),
            'lbp_svm': (lbp_svm, lbp_svm_scaler),
            'lbp_knn': (lbp_knn, lbp_knn_scaler),
            'lbp_rf': (lbp_rf, lbp_rf_scaler),
            'sift_svm': (sift_svm, sift_svm_scaler),
            'sift_knn': (sift_knn, sift_knn_scaler),
            'sift_rf': (sift_rf, sift_rf_scaler),
            'ensemble': (ensemble, None)
        }
        
        # Test all models on complex scenes
        complex_predictions = classifier.test_on_complex_scenes(
            complex_scenes_path, models_to_test, output_dir=output_dir
        )
        
        print("\nComplex scenes testing completed. Results saved to output directory.")


if __name__ == "__main__":
    main()
