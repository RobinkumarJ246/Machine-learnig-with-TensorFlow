import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputtxt = input("Enter your feedback : ")
input_sentence = inputtxt

input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)

input_tensor = tf.constant(input_ids)[None, :]

outputs = model(input_tensor)
predicted_label = tf.argmax(outputs.logits, axis=1).numpy()[0]

if predicted_label == 0:
    print("Thank you for your feedback")
else:
    print("We're extremely sorry for making you inconvenient! Thank you for your feedback")
