# Turkish Offensive Language Detection Web Application

This web application is designed to detect offensive language in Turkish texts using machine learning techniques. The application leverages Support Vector Machine (SVM) for classification and Flask for the web interface. Although the primary focus is on the Turkish language, the underlying model and approach can be adapted for other languages by using corresponding datasets and language-specific preprocessing steps.

By simply updating the dataset and adjusting the preprocessing techniques (such as stopword removal, tokenization, and text vectorization), this application can be extended to work with other languages, making it versatile for global use in detecting offensive language across multiple languages.

## Features
- Real-time offensive language detection
- Web-based user interface
- Turkish text preprocessing
- Machine learning classification
- TF-IDF vectorization

## Text Preprocessing
- Lowercase conversion
- Special character removal
- Number removal
- Turkish stop words removal
- Whitespace normalization

## Web Interface Features
- Clean and simple design
- Real-time analysis
- Color-coded results
- Responsive layout
- User-friendly interface

## Technologies Used
- Python 3.x
- Flask
- Scikit-learn
- Pandas
- HTML/CSS
- Pickle for model serialization

## Project Structure
```tree
content-moderation-ml-tr/
    ├── data/                    # Data directory
    │   ├── test.csv            # Test data
    │   ├── train.csv            # Training data
    │   └── valid.csv            # Validation data
    ├── html/
        └── index.html           # Web interface
    ├── pretrained_models/        # Pre-trained models directory
    │   ├── model.pkl             # Pre-trained model
    │   └── vectorizer.pkl        # Pre-trained vectorizer
    ├── app.py                     # Flask application
    └── save_model.py             # Model training script
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/mchtyldzx/content-moderation-ml-tr
```

2. Install required packages
```bash
pip install flask pandas scikit-learn
```

3. Choose one option:

### Option 1: Train Your Own Model (Recommended)
```bash
python save_model.py
```
This will train a new model with default parameters. You can modify the following parameters in save_model.py for better results:
- TF-IDF parameters (ngram_range, max_features, min_df, max_df)
- SVM parameters (C, kernel, gamma, class_weight)
- Text preprocessing options (stop words, special characters, numbers removal, etc.)

### Option 2: Use Pre-trained Model
Simply copy the files from pretrained_models/ to your root directory:
```bash
cp pretrained_models/* .
```

4. Run the application
```bash
python app.py
```

5. Open in browser
```bash
http://127.0.0.1:5000
```
Note: For better results specific to your use case, I recommend training your own model using save_model.py


### Default Model Parameters
- Algorithm: SVM (Support Vector Machine)
- Kernel: RBF
- C: 20
- gamma: 0.1
- class_weight: {0:1, 1:7}
- cache_size: 2000
- random_state: 42

### Default Vectorizer Parameters
- ngram_range: (1, 4)
- max_features: 8000
- min_df: 1
- max_df: 0.85
- sublinear_tf: True
- use_idf: True
- smooth_idf: True

## Model Performance
- Overall Accuracy: 83.83%
- Offensive Language Detection:
  * Precision: 0.68
  * Recall: 0.38
  * F1-score: 0.49
- Normal Text Detection:
  * Precision: 0.86
  * Recall: 0.95
  * F1-score: 0.90

## Test Data Results
The test data contains Turkish texts labeled as offensive language (1) or normal text (0):
- Total texts: 8,851
- Offensive language texts: 3,940 (44.51%)
- Normal texts: 4,911 (55.49%)
- Average text length (normal): 95.11 characters
- Average text length (offensive language): 113.65 characters
- Average word count (normal): 13.06 words
- Average word count (offensive language): 15.30 words


## Usage
1. Start the Flask application
2. Access the web interface
3. Enter Turkish text in the input field
4. Click "Analyze" button
5. View the classification result
6. Results are displayed as:
   - Green: Normal text
   - Red: Offensive language detected
     

## Acknowledgments
- [Dataset source](https://huggingface.co/datasets/Toygar/turkish-offensive-language-detection/tree/main)
