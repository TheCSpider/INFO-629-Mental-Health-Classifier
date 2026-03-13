# ----------------------------
# Import Classifier fucntions
# ----------------------------
import enum
import pickle
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer

class MentalHealthClassifiers(enum.Enum):
    NAIVE_BAYES = "Naive Bayes"
    NEURAL_NETWORK = "Neural Network"

class ClassifierLabels(enum.Enum):
    ANXIETY = "Anxiety"
    DEPRESSION = "Depression"
    NORMAL = "Normal"
    SUICIDAL = "Suicidal"
    ERROR = "ERROR"

def classify_mental_health(classifier: MentalHealthClassifiers, text: str) -> ClassifierLabels:
    if classifier == MentalHealthClassifiers.NAIVE_BAYES:
        return classify_naive_bayes(text)
    elif classifier == MentalHealthClassifiers.NEURAL_NETWORK:
        return classify_neural_network(text)
    else:
        return ClassifierLabels.ERROR
    
def classify_naive_bayes(text: str) -> ClassifierLabels:
    text = text.lower()
    
    # Load the pre‑trained Naive Bayes model
    with open('./naive_bayes/naive_bayes_model.pkl', 'rb') as file:
        nb_model = pickle.load(file)

    with open('./naive_bayes/count_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    
    query = [text]
    query_vector = vectorizer.transform(query)
    prediction = nb_model.predict(query_vector)
        
    return ClassifierLabels(prediction[0])

def classify_neural_network(text: str) -> ClassifierLabels:
    text = text.lower()

    # Load the pre-trained neural network model
    nn_model = tf.keras.models.load_model('./neural_network/neural_network.model.h5')

    # Load the corresponding CountVectorizer
    with open('./neural_network/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('./neural_network/label_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    # Transform the input text
    query = [text]
    query_vector = vectorizer.transform(query).toarray()

    # Get prediction from the neural network model
    prediction = nn_model.predict(query_vector)

    # Map prediction to ClassifierLabels enum
    predicted_label = encoder.inverse_transform([prediction.argmax()])[0]
    return ClassifierLabels(predicted_label)
