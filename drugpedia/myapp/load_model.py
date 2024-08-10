import os
import warnings
import joblib
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Suppress all other warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_joblib(file_name):
    file_path = os.path.join(BASE_DIR, 'models', file_name)
    return joblib.load(file_path)

# Load models
condition_model = load_model(os.path.join(BASE_DIR, 'models/condition_prediction.h5'))
rating_model = load_model(os.path.join(BASE_DIR, 'models/rating_predict.h5'))
sentiment_model = load_model(os.path.join(BASE_DIR, 'models/sentiment_model.h5'))
drug_exploration_model = load_model(os.path.join(BASE_DIR, 'models/drug_exploration.h5'))
lda_model = load_joblib('lda_model.joblib')
nmf_model = load_joblib('nmf_model.joblib')

# Load vectorizers and encoders using joblib
tfidf_vectorizer_condition = load_joblib('tfidf_vectorizer_condition.pkl')
count_vectorizer_condition = load_joblib('count_vectorizer_condition.pkl')
tfidf_vectorizer_rating = load_joblib('tfidf_vectorizer_rating.pkl')
count_vectorizer_rating = load_joblib('count_vectorizer_rating.pkl')
tokenizer = load_joblib('tokenizer.pkl')
tokenizer_explore = load_joblib('tokenizer_explore.pkl')
combined_vectorizer = load_joblib('combined_vectorizer.pkl')
arima_model = load_joblib('arima_model.pkl')
vectorizer = load_joblib('vectorizer.joblib')

# Load label encoders using joblib
label_encoder_condition = load_joblib('label_encoder_condition.pkl')
label_encoder = load_joblib('label_encoder.pkl')
le_condition = load_joblib('le_condition.pkl')
le_drug = load_joblib('le_drug.pkl')

print('All models and vectorizers loaded successfully')
