import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle


train_data = pd.read_csv('data/train.csv')
valid_data = pd.read_csv('data/valid.csv')
test_data = pd.read_csv('data/test.csv')


tr_stop_words = set([
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz',
    'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
    'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu',
    'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki',
    'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
])

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join([word for word in text.split() if word not in tr_stop_words])
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''


print("Cleaning texts...")
train_data['text'] = train_data['text'].apply(clean_text)
valid_data['text'] = valid_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)


print("Extracting features...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=8000,
    min_df=1,
    max_df=0.85,
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True
)

X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']


print("Training model...")
svc = SVC(
    C=20,
    kernel='rbf',
    gamma=0.1,
    class_weight={0:1, 1:7},
    random_state=42,
    cache_size=2000
)
model = svc.fit(X_train, y_train)


print("Saving model and vectorizer...")
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Process completed!")