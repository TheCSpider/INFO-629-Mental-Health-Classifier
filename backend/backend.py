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

    # Weighted "neural network–style" scoring
    weights = {
        ClassifierLabels.ANXIETY: {
            "panic": 2, "anxious": 3, "worry": 1, "overwhelmed": 2
        },
        ClassifierLabels.DEPRESSION: {
            "sad": 2, "empty": 3, "tired": 1, "hopeless": 3, "lost": 2
        },
        ClassifierLabels.SUICIDAL: {
            "suicide": 5, "kill myself": 5, "end it": 4, "die": 3
        },
        ClassifierLabels.NORMAL: {
            "happy": 2, "good": 1, "fine": 1, "okay": 1
        }
    }

    scores = {label: 0 for label in weights}

    # Compute weighted scores
    for label, word_dict in weights.items():
        for word, weight in word_dict.items():
            if word in text:
                scores[label] += weight

    # Pick the highest‑scoring label
    best_label = max(scores, key=scores.get)

    # If all scores are zero, assume NORMAL
    if scores[best_label] == 0:
        return ClassifierLabels.NORMAL
    return ClassifierLabels.DEPRESSION
    
classify_mental_health(MentalHealthClassifiers.NAIVE_BAYES,
                       "I feel hopeless and tired every day.")

classify_mental_health(MentalHealthClassifiers.NEURAL_NETWORK,
                       "I can't stop panicking and overthinking.")
