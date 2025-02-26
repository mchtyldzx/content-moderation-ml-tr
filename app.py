from flask import Flask, render_template, request, flash
import pickle
import re
import os
import sys

app = Flask(__name__)
model, vectorizer = None, None

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model_from_path(model_path, vectorizer_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()


def load_pretrained_model():
    return load_model_from_path(
        os.path.join(APP_DIR, 'pretrained_models', 'model.pkl'),
        os.path.join(APP_DIR, 'pretrained_models', 'vectorizer.pkl')
    )


def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

def initialize_model():
    global model, vectorizer
    model_path = os.path.join(APP_DIR, 'model.pkl')
    vectorizer_path = os.path.join(APP_DIR, 'vectorizer.pkl')

    model, vectorizer = load_model_from_path(model_path, vectorizer_path)

    if model is None or vectorizer is None:
        print("\nModel or vectorizer not found in the current directory.")
        print("Would you like to use the pre-trained models from the 'pretrained_models' directory? (y/n)")

        while True:
            choice = input("\nYour choice: ").lower()
            if choice == 'y':
                print("\nUsing pre-trained model and vectorizer from the 'pretrained_models' directory.")
                model, vectorizer = load_pretrained_model()
                if model is not None and vectorizer is not None:
                    break
                else:
                    print("Pre-trained model files not found. Please train a new model first using save_model.py or download them.")
                    print("You can download the model files from my repository: https://github.com/mchtyldzx/content-moderation-ml/pretrained_models")
                    print("After downloading, place them in the 'pretrained_models' directory and restart the application.")
                    sys.exit()
            elif choice == 'n':
                print("\nGreat choice!")
                print("To train your own model, run the following command: python save_model.py")
                print("After training, restart the application.")
                sys.exit()
            else:
                print("Invalid input. Please enter 'y' for the pre-trained model or 'n' to train your own model.")


initialize_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']

        cleaned_text = clean_text(text)
        text_vector = vectorizer.transform([cleaned_text])

        result = model.predict(text_vector)[0]

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
