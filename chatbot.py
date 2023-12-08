import tkinter as tk
import string
import random
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.tokenize import word_tokenize
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QLineEdit, QPushButton, QVBoxLayout, QWidget

nltk.download("punkt")
nltk.download("wordnet")

data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "La forme?", "yo", "Salut", "ça roule?", "Salut"],
            "responses": ["Salut à toi!", "Hello", "Comment vas-tu?", "Salutations!", "Enchanté", "Salut ! Comment puis-je vous aider aujourd'hui ?"]
        },
        {
            "tag": "age",
            "patterns": ["Quel âge as-tu?", "C'est quand ton anniversaire?", "Quand es-tu né?"],
            "responses": ["J'ai 25 ans", "Je suis né en 1996", "Ma date d'anniversaire est le 3 juillet et je suis né en 1996", "03/07/1996"]
        },
        {
            "tag": "date",
            "patterns": ["Que fais-tu ce week-end?", "Tu veux qu'on fasse un truc ensemble?", "Quels sont tes plans pour cette semaine"],
            "responses": ["Je suis libre toute la semaine", "Je n'ai rien de prévu", "Je ne suis pas occupé"]
        },
        {
            "tag": "name",
            "patterns": ["Quel est ton prénom?", "Comment tu t'appelles?", "Qui es-tu?"],
            "responses": ["Mon prénom est Miki", "Je suis Miki", "Miki"]
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "Salut", "see ya", "adios", "cya"],
            "responses": ["C'était sympa de te parler", "à plus tard", "On se reparle très vite!"]
        },
        {
            "tag": "probleme",
            "patterns": ["Mon wifi ne fonctionne pas"],
            "responses": ["Redémarrez votre routeur et votre ordinateur."]
        },
        {
            "tag": "demarrage",
            "patterns": [
                "Mon ordinateur ne démarre pas.",
                "Mon PC ne s'allume pas.",
                "Problème de démarrage de l'ordinateur."
            ],
            "responses": [
                "Assurez-vous que l'alimentation est branchée.",
                "Vérifiez les connexions électriques.",
                "Essayez de redémarrer l'ordinateur en maintenant le bouton d'alimentation enfoncé pendant quelques secondes."
            ]
        },
        {
            "tag": "virus",
            "patterns": [
                "Comment supprimer un virus de mon ordinateur ?",
                "Mon ordinateur est infecté par un malware.",
                "Besoin d'aide pour éliminer un virus."
            ],
            "responses": [
                "Utilisez un logiciel antivirus pour scanner et supprimer les virus.",
                "Mettez à jour votre logiciel antivirus pour une meilleure protection.",
                "Évitez de télécharger des fichiers suspects ou de cliquer sur des liens non fiables."
            ]
        },
        {
            "tag": "wifi",
            "patterns": [
                "Probleme",
                "Mon Wi-Fi ne fonctionne pas.",
                "Comment résoudre les problèmes de connexion Wi-Fi ?",
                "Pas de connexion Wi-Fi sur mon ordinateur."
            ],
            "responses": [
                "Donnez-moi plus de détails pour que je puisse vous aider.",
                "Redémarrez votre routeur et votre ordinateur.",
                "Vérifiez que vous êtes connecté au bon réseau Wi-Fi.",
                "Essayez de réinitialiser les paramètres réseau de votre ordinateur."
            ]
        },
        {
            "tag": "erreur",
            "patterns": [
                " "
            ],
            "responses": [
                "Something went wrong, please try reloading the conversation."
            ]
        },
        {
            "tag": "data",
            "patterns": [
                "Mon ordinateur ne démarre pas.",
                "Mon PC ne s'allume pas.",
                "Comment supprimer un virus de mon ordinateur ?",
                "Comment réparer un écran bleu sur mon ordinateur ?",
                "Comment récupérer des fichiers perdus sur mon disque dur ?",
                "Comment augmenter la vitesse de mon ordinateur ?",
                "Comment résoudre les problèmes de son sur mon ordinateur ?",
                "Comment nettoyer mon clavier d'ordinateur ?",
                "Comment protéger mon ordinateur contre les logiciels malveillants ?",
                "Comment résoudre les problèmes d'impression sur mon ordinateur ?",
                "Comment désinstaller un programme sur mon ordinateur ?",
                "Mon navigateur internet se bloque fréquemment.",
                "Comment récupérer un mot de passe oublié sur mon compte utilisateur ?",
                "Comment mettre à jour les pilotes de mon ordinateur ?",
                "Mon écran d'ordinateur affiche des lignes colorées.",
                "Comment vérifier l'état de santé de mon disque dur ?",
                "Comment empêcher mon ordinateur de surchauffer ?",
                "Comment changer le fond d'écran de mon ordinateur ?",
                "Mon clavier d'ordinateur ne fonctionne pas correctement."
            ],
            "responses": [
                "Assurez-vous que l'alimentation est branchée. Vérifiez les connexions électriques. Essayez de redémarrer l'ordinateur en maintenant le bouton d'alimentation enfoncé pendant quelques secondes.",
                "Vérifiez que l'alimentation est branchée et que le câble d'alimentation est en bon état. Essayez de débrancher et de rebrancher le câble d'alimentation. Si le problème persiste, contactez un technicien informatique.",
                "Utilisez un logiciel antivirus pour scanner et supprimer les virus. Mettez à jour votre logiciel antivirus régulièrement pour une meilleure protection. Évitez de télécharger des fichiers ou des logiciels provenant de sources non fiables.",
                "Un écran bleu peut être causé par plusieurs problèmes. Essayez de redémarrer votre ordinateur. Si le problème persiste, notez le code d'erreur affiché à l'écran et recherchez des solutions en ligne. Si nécessaire, contactez un technicien informatique.",
                "Utilisez un logiciel de récupération de données pour tenter de récupérer les fichiers perdus. Évitez d'écrire de nouvelles données sur le disque dur pour augmenter les chances de récupération réussie.",
                "Supprimez les fichiers inutiles et les programmes indésirables. Mettez à jour les pilotes de votre matériel. Ajoutez de la mémoire RAM si nécessaire. Évitez d'exécuter trop de programmes en même temps.",
                "Vérifiez que les haut-parleurs ou les écouteurs sont correctement connectés. Assurez-vous que le volume du système et du programme est suffisamment élevé. Mettez à jour les pilotes audio si nécessaire.",
                "Éteignez l'ordinateur et débranchez le clavier. Utilisez un chiffon doux et sec pour essuyer les touches. Utilisez un peu d'alcool isopropylique pour enlever les taches tenaces.",
                "Installez un logiciel antivirus de confiance et mettez-le à jour régulièrement. Évitez de télécharger des logiciels piratés ou provenant de sources non fiables. Soyez prudent avec les e-mails et les pièces jointes provenant d'expéditeurs inconnus.",
                "Vérifiez que l'imprimante est correctement connectée et alimentée. Assurez-vous que le pilote d'imprimante est installé et à jour. Réessayez d'imprimer ou redémarrez l'imprimante si nécessaire.",
                "Allez dans le Panneau de configuration, puis dans Programmes et fonctionnalités. Sélectionnez le programme que vous souhaitez désinstaller et cliquez sur 'Désinstaller'.",
                "Effacez le cache et les cookies de votre navigateur. Mettez à jour votre navigateur vers la dernière version. Si le problème persiste, essayez d'utiliser un autre navigateur.",
                "Si vous avez configuré une option de récupération de mot de passe (comme une adresse e-mail de secours ou un numéro de téléphone), utilisez-la pour récupérer votre mot de passe. Sinon, contactez le support technique du service concerné.",
                "Allez dans le Gestionnaire de périphériques, sélectionnez le périphérique dont vous souhaitez mettre à jour le pilote, cliquez avec le bouton droit de la souris et choisissez 'Mettre à jour le pilote'.",
                "Ce problème peut être causé par un mauvais contact entre l'écran et l'ordinateur, ou par un dysfonctionnement du matériel. Vérifiez les connexions et contactez un technicien si nécessaire.",
                "Utilisez un logiciel de diagnostic de disque dur pour vérifier l'état de santé de votre disque dur. Recherchez les erreurs et les secteurs défectueux.",
                "Assurez-vous que les ventilateurs de refroidissement de l'ordinateur fonctionnent correctement. Évitez de bloquer les évents d'aération. Nettoyez régulièrement la poussière accumulée à l'intérieur de l'ordinateur.",
                "Faites un clic droit sur le bureau, puis choisissez 'Personnaliser'. Sélectionnez l'image que vous souhaitez utiliser comme fond d'écran.",
                "Vérifiez que le clavier est correctement connecté à l'ordinateur. Redémarrez l'ordinateur. Si le problème persiste, essayez d'utiliser un autre clavier."
            ]
        }
    ]
}

lemmatizer = WordNetLemmatizer()
words = []
classes = []
doc_X = []
doc_y = []


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))
print(words)

training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))

adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
print(model.summary())

# entraînement du modèle
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_intent(user_input, vocab, intents_data):
    tokens = clean_text(user_input)
    bow = bag_of_words(" ".join(tokens), vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    intents = [(i, res) for i, res in enumerate(result) if res > thresh]
    intents.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for i, res in intents:
        tag_index = i
        tag = classes[tag_index]
        return_list.append(tag)
    return return_list

def get_response(intents_list, intents_data):
    for tag in intents_list:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Je ne comprends pas cette question, pouvez-vous reformuler?"

class ChatbotGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Chatbot')
        self.setGeometry(100, 100, 800, 600)

        self.chat_history = QTextBrowser(self)
        self.chat_history.setGeometry(20, 20, 760, 400)

        self.user_input = QLineEdit(self)
        self.user_input.setGeometry(20, 440, 600, 30)

        self.send_button = QPushButton('Send', self)
        self.send_button.setGeometry(640, 440, 140, 30)
        self.send_button.clicked.connect(self.send_message)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addWidget(self.user_input)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.chat_history.append('Chatbot: Hello! How can I help you?')

    def send_message(self):
        user_message = self.user_input.text()
        self.user_input.clear()

        intents = pred_intent(user_message, words, classes)
        response = get_response(intents, data)

        self.chat_history.append('You: ' + user_message)
        self.chat_history.append('Chatbot: ' + response)
        

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = ChatbotGUI()
    window.show()
    sys.exit(app.exec_())  