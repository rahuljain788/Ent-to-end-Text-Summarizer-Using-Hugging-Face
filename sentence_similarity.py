from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
print("********************************")
model_name = "bert-base-cased-finetuned-mrpc"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sequence_0 = "50% off on "
sequence_1 = "The problem was solved by a young mathematician."
sequence_2 = "The Sky is Blue and beautiful"

tokens = tokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
classification_logits = model(**tokens)[0]
results = torch.softmax(classification_logits, dim=1).tolist()[0]

classes = ["not paraphrase", "is paraphrase"]
for i in range(len(classes)):
    print(f"{classes[i]}: {round(results[i] * 100)}%")