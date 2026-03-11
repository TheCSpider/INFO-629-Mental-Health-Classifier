# ----------------------------
# Import Classifier fucntions
# ----------------------------
import enum
import pickle

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
    # text = text.lower()

    # anxiety_words = ["worried", "anxious", "panic", "nervous", "overthinking"]
    # depression_words = ["sad", "empty", "tired", "hopeless", "worthless"]
    # suicidal_words = ["suicide", "kill myself", "end it", "no reason to live"]
    # normal_words = ["happy", "good", "fine", "okay", "normal"]

    # scores = {
    #     ClassifierLabels.ANXIETY: sum(w in text for w in anxiety_words),
    #     ClassifierLabels.DEPRESSION: sum(w in text for w in depression_words),
    #     ClassifierLabels.SUICIDAL: sum(w in text for w in suicidal_words),
    #     ClassifierLabels.NORMAL: sum(w in text for w in normal_words)
    # }

    # # Pick the label with the highest score
    # best_label = max(scores, key=scores.get)

    # # If no keywords matched, default to NORMAL
    # if scores[best_label] == 0:
    #     return ClassifierLabels.NORMAL

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
    with open('./neural_network/neural_network_model.pkl', 'rb') as file:
        nn_model = pickle.load(file)

    # Load the corresponding CountVectorizer
    with open('./neural_network/count_vectorizer.pkl', 'rb') as file:
        vectorizer: CountVectorizer = pickle.load(file)

    # Transform the input text
    query = [text]
    query_vector = vectorizer.transform(query)

    # Get prediction from the neural network model
    prediction = nn_model.predict(query_vector)

    # Map prediction to ClassifierLabels enum
    return ClassifierLabels(prediction[0])

  
classify_mental_health(MentalHealthClassifiers.NAIVE_BAYES,
                       "I feel hopeless and tired every day.")

classify_mental_health(MentalHealthClassifiers.NEURAL_NETWORK,
                       "I can't stop panicking and overthinking.")
