

!pip install datasets>=2.18.0 transformers>=4.38.2 accelerate>=0.27.2 evaluate
from datasets import load_dataset
import random

all_data = load_dataset("CreativeLang/ColBERT_Humor_Detection").filter(lambda example: len(example["text"])<1024)["train"]
random.seed(1)

# Load our small sample of data for training and analysis
random_train_indices = random.sample(range(len(all_data)), 18000)
train_data = all_data.select(random_train_indices)

random_eval_indices = random.sample(range(len(all_data)), 2000)
eval_data = all_data.select(random_eval_indices)

random_test_indices = random.sample(range(len(all_data)), 2000)
test_data = all_data.select(random_test_indices)

"""<h1>Classifier 1</h1>
<i>Classifier 1: Rule-based. Write a rule-based classifier with regular expressions</i>
"""

import re

def regex_classifier(text):
    humor_patterns = [
        r'^(Q:|Question:).*(A:|Answer:)',
        r'[.!?]$',
        r'\bwhat do you\b',
        r'\bwhy did\b',
        r'\bwhy does\b',
        r'who\s+.*'
    ]

    for pattern in humor_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 1
    return 0

for i in range(100):
    sample = eval_data[i]

    text = sample["text"]
    true_label = sample["humor"]

    predicted = regex_classifier(text)

    print(f"=== Sample {i + 1} Analysis ===")
    print("Text: ", text)
    print("True Label: ", "TRUE" if true_label else "FALSE")
    print("Predicted Label: ", "True" if predicted else "False")
    print("========================\n")

for i in range(20):
  print(train_data[i])

# evaluation

def recall(true_positives, false_negatives):
  return true_positives / (true_positives + false_negatives)

def precision(true_positives, false_positives):
  return true_positives / (true_positives + false_positives)

def accuracy(true_positives, true_negatives, false_positives, false_negatives):
  return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

def f1_score(precision, recall):
  return (2 * precision * recall) / (precision + recall)

def evaluate(data):
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  pos = 0
  neg = 0
  for i in range(len(data)):
    if data[i]["humor"]  == True and regex_classifier(data[i]["text"]) == True:
      tp += 1
    elif data[i]["humor"]  == False and regex_classifier(data[i]["text"]) == False:
      tn += 1
    elif data[i]["humor"]  == False and regex_classifier(data[i]["text"]) == True:
      fn += 1
    else:
      fp += 1

  for i in range(len(data)):
    if data[i]["humor"] == True:
      pos += 1
    else:
      neg += 1

  rec = recall(tp, fn)
  prec = precision(tp, fp)
  acc = accuracy(tp, tn, fp, fn)
  f1 = f1_score(prec, rec)

  pos_per = (pos / (len(data) - 1)) * 100
  neg_per = (neg / (len(data) - 1)) * 100

  print("Positive Joke %:", pos_per)
  print("Negative Joke %:", neg_per)

  print("Recall:", rec)
  print("Precision:", prec)
  print("Accuracy:", acc)
  print("F1 Score:", f1_score(prec, rec))

# evaluation output
evaluate(test_data)

"""**Design of the Rule-Based Humor Classifier**

This rule-based classifier uses regular expressions (regex) to detect patterns commonly associated with humorous text. It flags a piece of text as humorous (1) if it matches any predefined regex pattern, and non-humorous (0) otherwise. The classifier is designed to look for linguistic cues and structures that are typical of jokes or humorous content.

**Regex Rules Used and Why:**

1.	r'^(Q:|Question:).*(A:|Answer:)' : It matches Q&A formats like “Q: Why did the chicken… A: To get to the other side,” which are classic joke setups.
2.	r'[.!?]$' : It ensures the sentence ends with punctuation, which is typical in well-formed, deliberate jokes (in contrast to casual or incomplete statements).
3.	r'\bwhat do you\b' : Captures joke structures like “What do you call a…” — a common joke opening.
4.	r'\bwhy did\b' and r'\bwhy does\b' : These phrases often signal the setup of a joke or a humorous question.
5.	r'who\s+.*' : Picks up on exaggerated or rhetorical questions like “Who even does that?” which may be used in a humorous or sarcastic tone.

These patterns were chosen because they represent common structure and cues used in jokes, especially in written text like tweets or stand-up scripts. While not foolproof, they give the classifier a simple logic to detect likely humor without needing machine learning.

**Evaluation**
1. For evaluation, we used accuracy, since both true positives and true negatives are important, and the dataset is roughly balanced. The classifier achieved **83.7%** accuracy, which is reasonable for a lightweight, rule-based approach using just six regex rules.

<h1>Classifier 2</h1>
<i>Classifier 2: Logistic Regression + TF-IDF (like we did in class for the imdb dataset)</i>
"""

# Classifier 2: Logistic Regression + TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(max_df=0.5)
vectorizer = TfidfVectorizer(max_df=0.5, ngram_range=(2, 2), min_df=1)
X_train = vectorizer.fit_transform(train_data["text"])
X_test = vectorizer.transform(test_data["text"])

model = LogisticRegression(verbose=1)
model.fit(X_train, train_data["humor"])

predicted_lr = model.predict(X_test)
print(classification_report(test_data["humor"], predicted_lr))

print(predicted_lr[0])

"""**Design of the Logistic Regression + TF-IDF Classifier**

**Hyperparameters:**
1. With the original hyperparameter we had an accuracy of 91%, however, when we included some hyperparameters such as ngram_range and min_dif there was a 5% decrease in accuracy.

2. We also observed that increasing max_df to 1 drastically decreases the accuracy down to 50% because it allowed overly common, non-informative bigrams into the feature set. These frequent terms added noise and diluted the discriminative power of the model, making it harder to distinguish between humorous and non-humorous text.

**Evaluation Metric**
1. For evaluation metric we decided to focus on the F1-score because it balances precision and recall; both essential for tasks like humor detection where false positives and false negatives carry different consequences.

2. The Logistic Regression + TF-IDF model achieved an accuracy of 86%. This is with the hyperparameters that we included. Compared to the Regex model, which has an accurary of 83.7%, this Logistic Regression + TF-IDF model is 2.3% more accurate. If we kept just the max_df hyperparameter, we would have had an accuracy of 91%. Regardless of the difference in the hyperparameters, the Logistic Regression + TF-IDF model has a higher accuracy than the Regex model.

# <h1>Classifier 3</h1>
<i>Classifier 3: Fine-tune an encoder model (like we did in class on the rotten tomatoes dataset)</i>
"""

# Classifier 3: Full fine-tuned encoder model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import evaluate

# Load Model and Tokenizer
model_id = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2) #attach a classification head on top of your model (one layer nn on top of your embeddings)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model)

from datasets import ClassLabel, Value
# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
   """Tokenize input data"""
   return tokenizer(examples["text"], truncation=True)

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)


tokenized_train = tokenized_train.rename_column("humor", "label")
print(type(tokenized_train[0]["label"]))

tokenized_test = tokenized_test.rename_column("humor", "label")

tokenized_train = tokenized_train.cast_column("label", Value("int64"))
tokenized_test = tokenized_test.cast_column("label", Value("int64"))

print(type(tokenized_train[0]["label"]))

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    """Calculate F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]

    """Calculate Accuracy"""
    #accuracy = accuracy_score(labels, predictions)
    load_accuracy = evaluate.load("accuracy")
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]

    return {"f1": f1, "accuracy": accuracy}

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=1,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

print(tokenized_train[0])
# text, humor, input_ids, token_type_ids, attention_mask

trainer.train()

trainer.evaluate()

"""### Freeze Layers"""

# Load Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print layer names
for name, param in model.named_parameters():
    print(name)

for name, param in model.named_parameters():

     # Trainable classification head
     if name.startswith("classifier"):
        param.requires_grad = True

      # Freeze everything else
     else:
        param.requires_grad = False

# We can check whether the model was correctly updated
for name, param in model.named_parameters():
     print(f"Parameter: {name} ----- {param.requires_grad}")

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)
trainer.train()

trainer.evaluate()

# find misclassifications
from numpy import argmax
def misclassifications(data):
  results = trainer.predict(tokenized_test)
  encoder_pred = argmax(results.predictions, axis=-1)
  lr_reg_counter = 0
  encoder_lr_counter = 0
  encoder_counter = 0

  print("Three sentences that the LR model gets correct and the regex model gets incorrect:")
  for i in range(len(data)):
    if (data[i]["humor"]  == True and predicted_lr[i] == True and regex_classifier(data[i]["text"]) == False) or (data[i]["humor"]  == False and predicted_lr[i] == False and regex_classifier(data[i]["text"]) == True):
      lr_reg_counter += 1
      print("Data:", data[i], ", LR:", predicted_lr[i], ", Regex:", regex_classifier(data[i]["text"]))

    if lr_reg_counter == 3:
      break

  print("Three sentences that the encoder model gets correct and the LR model gets incorrect:")
  for i in range(len(data)):
    if (data[i]["humor"]  == True and encoder_pred[i] == True and predicted_lr[i] == False) or (data[i]["humor"]  == False and encoder_pred[i] == False and predicted_lr[i] == True):
      encoder_lr_counter += 1
      print("Data:", data[i], ", Encoder:", encoder_pred[i], ", LR:", predicted_lr[i])

    if encoder_lr_counter == 3:
      break

  print("Three sentences that our best encoder model gets incorrect:")
  for i in range(len(data)):
    if (data[i]["humor"]  == True and encoder_pred[i] == False) or (data[i]["humor"]  == False and encoder_pred[i] == True):
      encoder_counter += 1
      print("Data:", data[i], ", Encoder:", encoder_pred[i])
    if encoder_counter == 3:
      break

misclassifications(test_data)

from numpy import argmax
def misclass(data):
  results = trainer.predict(tokenized_test)
  encoder_pred = argmax(results.predictions, axis=-1)
  for i in range(len(data)):
    if (data[i]["humor"]  == True and encoder_pred[i] == False) or (data[i]["humor"]  == False and encoder_pred[i] == True):
      print("Data:", data[i], ", Encoder:", encoder_pred[i])

misclass(test_data)

"""## **Error Analysis**

*For each set of examples, write your best analysis (20-50 words) of why you think the model got it wrong. Your analysis should explore if there's a general pattern to the type of sentences each classifier gets wrong on this task.*


1. **3 examples of sentences from eval that Logistic Regression+TF-IDF gets right but regex rules gets wrong.**

Sentence 1:
“How did Cosby fuck up his phone? he put it on sleep mode”

**Analysis:**
The regex model misses this joke because it doesn't match any predefined patterns. It lacks understanding of wordplay or cultural references. In contrast, the TF-IDF model likely recognizes humor patterns in the phrasing or keyword associations like “fuck up,” “phone,” and “sleep mode.”

Sentence 2:
“Why don't elephants smoke? they can't fit their butts in the ashtray”

**Analysis:**
Although the joke starts with “Why don't…”, which resembles a pattern the regex is designed for, it likely fails due to strict spacing or token structure. Logistic Regression handles this better by learning from similar bi-gram patterns like “elephants smoke” or “butts ashtray.”

Sentence 3:
“Should police be allowed to keep property without a criminal conviction?”

**Analysis:**
The regex wrongly classifies this as humorous due to the “should [subject]” format resembling a joke pattern. However, the sentence is serious. Logistic Regression correctly identifies the tone as non-humorous by considering word context like “police,” “conviction,” and “property.”



2. **3 examples of sentences from eval that encoder gets right but Logistic Regression gets wrong.**

Sentence 1:
"It's complicated relationship status = someone cheated but we signed a lease."

**Analysis:**
The Encoder model classified the text as humorous because it understands the overall sentence structure and underlying contradiction. It captures the emotional implication of “cheated” versus the mundane “lease”, which creates comedic contrast.
The Logistic Regression classified the text as not humorous because it only sees surface-level bigrams like “signed lease” or “relationship status”, which aren’t humorous on their own and are often found in serious or neutral contexts. Unlike the Encoder model, it can’t understand the deeper irony that comes from connecting different parts of the text.

Sentence 2:'Bought a sled on sale in boston got a real tobahgain.'

**Analysis:**
This text was classified as humor by the Encoder model because it has been trained on a wide range of language and can understand that “tobahgain” is a funny misspelling based on the context. Its ability to understand meaning and context helps it spot the pun and recognize it as humor.
The Logistic Regression classified the text as not humorous because it only looks at simple word patterns and how often words appear together. It doesn’t understand things like context, wordplay, or language differences. The model doesn’t recognize the Boston accent pun (“tobahgain” for “bargain”) or the reference to “toboggan” as a type of sled, missing the irony in the joke. Since “tobahgain” is an uncommon spelling not seen often in its training data, the Logistic Regression model failed to recognize it as a humorous variation.

Sentence 3:"Plan ahead - it wasn't raining when noah built the ark."

**Analysis:**
This text was classified as humor by the Encoder model because the model was likely familiar with the story of Noah’s Ark. This helps it pick up on the idea of planning ahead for something that seems unlikely or far off, and it understands the humor in how that serious preparation is described in a casual or understated way. The Logistic Regression classified the text as not humorous because it likely does not recognize “Noah’s ark” as a well-known story about preparing for a future disaster. It might see “raining” and “built” as separate events without the necessary background knowledge. The model treats “plan ahead”, “noah built”, or “raining when” as isolated bigrams, none of which seem funny without the full background


3. **3 examples of sentences from eval that your best encoder model still gets wrong.**

Sentence 1:
"Nobody's going to see a sticker on a telephone pole and then become a fan of your band."

**Analysis:**
The encoder classifies this sentence incorrectly. It classifies it as not a joke when it is in fact a joke. Since the humor is dry, observational humor, the model is struggling to identify it as a joke. Also it does not follow the typical structure of a joke, making it more challenging for the model to classify correctly.

Sentence 2:
"Talking to marco rubio: a scripted candidate suddenly gets chatty"

**Analysis:**
The encoder classifies this sentence incorrectly. It classifies it as a joke when it is not a joke. With the use of the colon, the structure is similar to some jokes, causing the misclassification. Also, there is irony that the encoder is viewing as humourous, and thus misclassifying it as a joke.

Sentence 3:
"Perfect house, perfect spouse? how finding your dream home is like dating"

**Analysis:**
The encoder classifies this sentence incorrectly. It classifies it as a joke when it is not a joke. The structure is question-answer, which is typical of a joke, and it compares unrelated domains which is also common in humor, causing the encoder to misclassify it as a joke.

<h1>Bonus:</h1>

Bonus (4 points to overall grade): *Zero-shot and few-shot evaluate a generative LM model on the test dataset, and achieve an accuracy greater than 90%.*
"""

!pip install -q transformers accelerate bitsandbytes datasets

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "MBZUAI/LaMini-Flan-T5-783M"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

FEW_SHOT = """
Instruction: Is the following text a joke? Respond only with "Yes" or "No".

Text: "Why don’t skeletons fight each other? They don’t have the guts."
Answer: Yes

Text: "The quarterly update will be shared tomorrow."
Answer: No

Text: "I'm on a seafood diet. I see food and I eat it!"
Answer: Yes

Text: "The system will be offline for maintenance at midnight."
Answer: No

Text: "What do you call fake spaghetti? An impasta!"
Answer: Yes

Text: "Please sign the attendance sheet at the front desk."
Answer: No
"""

def make_few_shot_prompt(text):
    return f"""{FEW_SHOT}

Text: "{text}"
Answer:"""

def make_zero_shot_prompt(text):
    return f"""Instruction: Is the following text a joke? Respond only with "Yes" or "No".
Text: "{text}"
Answer:"""

def extract_yes_no_only(output_text):
    output_text = output_text.lower()
    if "yes" in output_text:
        return 1
    elif "no" in output_text:
        return 0
    return 0  # fallback to No

def mini_flan_few_shot_predict(text):
    prompt = make_few_shot_prompt(text)
    result = generator(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_p=0.9,
        do_sample=False
    )[0]["generated_text"]
    return extract_yes_no_only(result)

def mini_flan_zero_shot_predict(text):
    prompt = make_zero_shot_prompt(text)
    result = generator(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_p=0.9,
        do_sample=False
    )[0]["generated_text"]
    return extract_yes_no_only(result)

def evaluate_model(predict_func, data,title="Model"):
    correct = 0
    num = len(data)
    for i in range(num):
        text = data[i]["text"]
        true = data[i]["humor"]
        pred = predict_func(text)
        if pred == true:
            correct += 1
        #print(f"[{i+1}] Pred: {pred} | True: {true} | Text: {text[:50]}...")

    acc = correct / num
    print(f"\n {title} Accuracy on {num} samples: {acc * 100:.2f}%")
    return acc

evaluate_model(mini_flan_zero_shot_predict, test_data, title="LaMini-Flan-T5 Zero-Shot")
evaluate_model(mini_flan_few_shot_predict, test_data, title="LaMini-Flan-T5 Few-Shot")

"""Bonus (1 point to overall grade): *Achieve an accuracy >= 98% on this task with your encoder model.*"""

# Load Model and Tokenizer
model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2) #attach a classification head on top of your model (one layer nn on top of your embeddings)
tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import ClassLabel, Value
# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
   """Tokenize input data"""
   return tokenizer(examples["text"], truncation=True)

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)


tokenized_train = tokenized_train.rename_column("humor", "label")
print(type(tokenized_train[0]["label"]))

tokenized_test = tokenized_test.rename_column("humor", "label")

tokenized_train = tokenized_train.cast_column("label", Value("int64"))
tokenized_test = tokenized_test.cast_column("label", Value("int64"))

print(type(tokenized_train[0]["label"]))

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()