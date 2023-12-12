import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
cntxt = input("Type a context : ")
qn = input("Type a question : ")
context = cntxt
question = qn
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="tf")
outputs = model(inputs)
start_scores, end_scores = outputs.start_logits, outputs.end_logits
start_index = tf.argmax(start_scores, axis=1).numpy()[0]
end_index = tf.argmax(end_scores, axis=1).numpy()[0] + 1
answer_tokens = inputs["input_ids"].numpy()[0][start_index:end_index]
answer = tokenizer.decode(answer_tokens)
print("Bot : ",answer)
