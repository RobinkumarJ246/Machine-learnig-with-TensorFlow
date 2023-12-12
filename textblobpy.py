from textblob import TextBlob

def generate_response(input):
    blob = TextBlob(input)
    
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0.5:
        response = "That sounds great! How can I assist you today?"
    elif sentiment > 0:
        response = "I'm glad to hear that! What can I help you with?"
    elif sentiment < -0.5:
        response = "I'm sorry to hear that. How can I help you feel better?"
    elif sentiment < 0:
        response = "I'm sorry to hear that. What can I do to help?"
    else:
        response = "I'm not sure I understand. Can you please rephrase your question or request?"
    
    return response

print("Type end to terminate")
while True:
    text = input("Type something : ")
    input_text = text
    response_text = generate_response(input_text)
    print(response_text)