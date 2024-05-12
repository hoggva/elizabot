class SimpleChatBot:
    def __init__(self, training_data):
        self.training_data = training_data
        self.vocabulary, self.weights = self.build_vocabulary_and_weights(training_data)

    def tokenize(self, text):
        return text.lower().split()

    def build_vocabulary_and_weights(self, training_data):
        vocabulary = set()
        weights = {}
        for question, response, keywords in training_data:
            tokens = self.tokenize(question)
            for token in tokens:
                vocabulary.add(token)
                if token in keywords:
                    weights[token] = keywords[token]
                else:
                    weights[token] = 1  # default weight
        return sorted(vocabulary), weights

    def text_to_vector(self, text):
        tokens = self.tokenize(text)
        vector = [0] * len(self.vocabulary)
        for token in tokens:
            if token in self.vocabulary:
                index = self.vocabulary.index(token)
                vector[index] = self.weights.get(token, 1)
        return vector

    def dot_product(self, vector_a, vector_b):
        return sum(a * b for a, b in zip(vector_a, vector_b))

    def respond(self, input_text):
        input_vector = self.text_to_vector(input_text)
        best_match = max(self.training_data, key=lambda data: self.dot_product(input_vector, self.text_to_vector(data[0])))
        return best_match[1]

# Example training data: list of tuples (input phrase, response, {keyword: weight})
training_data = [
    ("hello", "Hi there! How can I assist you today?", {"hello": 2}),
    ("what is your name", "I'm a Python chatbot.", {"name": 2}),
    ("how are you", "I'm just a program, so I'm always doing well!", {"how": 1, "you": 1}),
    ("goodbye", "Goodbye! Have a great day!", {"goodbye": 2})
]

# Create an instance of the chatbot
chatbot = SimpleChatBot(training_data)

# Interacting with the chatbot
print("Welcome to Chatbot!")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = chatbot.respond(user_input)
    print("Chatbot:", response)

