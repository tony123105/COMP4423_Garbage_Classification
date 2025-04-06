# Garbage Classification using Traditional Machine Learning

## Project Overview

This project implements and evaluates traditional machine learning approaches for classifying garbage into 10 distinct categories: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash[cite: 2]. The goal is to explore the effectiveness of different feature extraction techniques and classification algorithms for this task, providing insights into the challenges and potential of non-deep learning methods for waste management[cite: 8, 9, 11, 14].

## Dataset

* **Source:** The dataset consists of images representing 10 common garbage categories[cite: 18].
* **Sampling:** To address potential training bias from class imbalance, the dataset was sampled to include 200 images per category, totaling 2000 images for training/testing[cite: 19, 20]. The original and sampled distributions can be seen in the generated `output/dataset_sampling_comparison.png` figure.
* **Preprocessing:**
    * Images were resized to 128x128 pixels[cite: 21].
    * Images were converted to RGB and then grayscale for feature extraction[cite: 22].
    * Pixel values were normalized[cite: 23].
* **Splitting:** The data was split into 80% for training (160 samples per class) and 20% for testing (40 samples per class)[cite: 24].

## Methodology

### Feature Extraction

Three feature extraction methods were implemented and compared[cite: 3, 26]:
1.  **Histogram of Oriented Gradients (HOG):** Captures shape-based patterns. Effective for items with distinct shapes like bottles, cans, and boxes[cite: 28, 29]. Visualized in `output/hog_feature_space.png`.
2.  **Local Binary Patterns (LBP):** Captures global texture patterns. Useful for items like paper, clothes, and biological waste[cite: 30, 34]. Visualized in `output/lbp_feature_space.png`.
3.  **Scale-Invariant Feature Transform (SIFT):** Detects local keypoints invariant to scale and rotation, using a Bag of Visual Words (BoVW) approach (100 visual words)[cite: 35, 36]. Effective for items with distinct local features or labels[cite: 38]. Visualized in `output/sift_feature_space.png`.

Feature visualizations for sample images are available in `output/feature_visualization.png`.

### Classification Models

Three classifiers were trained and evaluated for each feature type[cite: 52]:
1.  **Support Vector Machine (SVM):** Optimized using GridSearchCV with various kernels, C values, and gamma settings[cite: 53].
2.  **K-Nearest Neighbors (KNN):** Optimized using GridSearchCV for K-values, distance metrics, and weighting[cite: 54, 55].
3.  **Random Forest (RF):** Optimized using GridSearchCV for the number of trees, maximum depth, and minimum samples per split[cite: 56, 57].

### Ensemble Method

* A soft voting ensemble classifier was created, combining the predictions (based on probability) of all nine individual models (3 features x 3 classifiers)[cite: 62]. All models were weighted equally[cite: 63].

## Key Results

* **Individual Models:** The HOG+SVM combination achieved the highest accuracy among individual models at 51.31% (cross-validation) [cite: 4, 186] and 54.75% (test set)[cite: 150].
* **Ensemble Model:** The ensemble approach significantly improved performance, achieving a test set accuracy of 63.25%[cite: 4, 135, 186].
* **Performance Details:** Detailed classification reports and confusion matrices for each model are available in the `output/` directory (e.g., `output/classification_report_Ensemble.txt`, `output/confusion_matrix_HOG+SVM.png`). A summary table is in `output/model_performance.csv`[cite: 186].
* **Complex Scenes:** Testing on complex, real-world scenes (e.g., cluttered kitchens, beach pollution) revealed significant challenges, including multi-object interference, background influence, and limitations of the feature extractors[cite: 5, 66, 89, 91, 92, 93]. Predictions for these scenes are summarized in `output/complex_scenes_predictions.csv`[cite: 185].

## How to Run

1.  **Prerequisites:** Ensure you have Python installed along with the following libraries:
    * `opencv-python`
    * `numpy`
    * `matplotlib`
    * `scikit-learn`
    * `scikit-image`
    * `pandas`
    *(You can usually install these using pip: `pip install opencv-python numpy matplotlib scikit-learn scikit-image pandas`)*
2.  **Dataset:** Place the image dataset in a directory named `reference_images`, with subdirectories for each class (e.g., `reference_images/paper/`, `reference_images/glass/`, etc.).
3.  **Complex Scenes (Optional):** Place complex scene test images in a directory named `complex_scenes`.
4.  **Execution:** Run the main script:
    ```bash
    python Assignment1_22080062d_LAI_KACHUNG.py
    ```
5.  **Output:** All results, including visualizations (feature spaces, confusion matrices, performance tables, complex scene analyses) and classification reports, will be saved in the `output/` directory.

## Discussion and Future Improvements

While the ensemble model shows promise, classification in complex, real-world scenes remains challenging for these traditional methods[cite: 109, 124]. Key challenges include handling multiple objects per image, background interference, and feature limitations[cite: 89, 112, 113, 114, 115].

Proposed improvements include[cite: 6]:
* **Scene Segmentation:** Implement object detection/segmentation to isolate items before classification[cite: 118].
* **Enhanced Feature Engineering:** Combine features (HOG, LBP, SIFT) and incorporate color information[cite: 104, 119].
* **Advanced Ensemble Methods:** Use weighted voting or stacking[cite: 120].
* **Data Augmentation:** Expand the training set with more varied images (backgrounds, lighting, multiple objects, occlusion)[cite: 105, 121].

This project highlights the capabilities and limitations of traditional machine learning for garbage classification, offering a baseline and interpretable alternative to deep learning approaches[cite: 7, 107, 127, 128].
