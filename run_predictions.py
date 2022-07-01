from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import csv
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Loading the dataset
infile = "but_inferred_propositions.csv"
outfile = "./results/results.csv"

with open(infile, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    content = [row for row in reader] # Saving all the rows as `content'
    pairs = [[row[2], row[4]] for row in content[1:]] # Extracting only the premises and hypotheses

    
# Load model
tokenizer = T5Tokenizer.from_pretrained("t5-large", force_download=True)
model = T5ForConditionalGeneration.from_pretrained("t5-large", force_download=True)

print('Loading...')

# Taken from https://huggingface.co/docs/transformers/model_doc/t5#inference
# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

start_time = time.time()

task_prefix = "mnli"
sentences = [" premise: " + pairs[i][0] + " hypothesis: " + pairs[i][1] for i in range(len(pairs))]
inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
end_time = time.time()
print(f'Time taken : {end_time-start_time}')


# Saving the output
with open(outfile, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile, delimiter=';',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(content[0] + ['prediction'])
    for i, row in enumerate(content[1:]):
        writer.writerow(row + [predictions[i]])

        
# Generating failure rate table

# Read the output
with open(outfile, 'r', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    content = [row for row in reader] # Saving all the rows as `content'

# Calculating failure rate    
proposition_type = ['undone_action', 'sentiment', 'world_knowledge', 'noun_relation']
alteration = ['and', 'negation', 'temporal', 'sentiment', 'reversed']    
    
# Compared with expected labels

failure_rate = defaultdict(dict)

for t in proposition_type:
    base_pred = []
    for row in content[1:]: # First loop for all base examples for this type
        if row[0] == t:
            if row[1] == "":
                base_pred.append(row[-1]) # Make a list of 'base' predictions
                success = base_pred.count('contradiction')
                failure_rate[t][''] = (len(base_pred) - success) / len(base_pred) * 100
    for a in alteration:
        nr_alt = 0
        nr_fail = 0
        for row in content[1:]: # Second loop for an alteration for this type
            if row[0] == t and row[1] == a:
                nr_alt += 1
                if row[3] != row[-1]: # Fail if prediction isn't equal to expected
                    nr_fail += 1
                failure_rate[t][a] = nr_fail / nr_alt * 100

df = pd.DataFrame(failure_rate).T.astype(float).round(2).fillna('/').to_string()
print("Failure rate compared with expected labels")
print(df)                    
                    
# Compared with base examples

failure_rate = defaultdict(dict)

for t in proposition_type:
    base_pred = []
    each_alt_pred = {}
    all_alt_pred = []
    for row in content[1:]: # Skip the header
        if row[0] == t:
            if row[1] == "":
                base_pred.append(row[-1]) # Make a list of 'base' predictions
                success = base_pred.count('contradiction')
                failure_rate[t][''] = (len(base_pred) - success) / len(base_pred) * 100
                if each_alt_pred:
                    all_alt_pred.append(each_alt_pred) # Make a list of all predictions for all alterations
                each_alt_pred = {} # Reset the dict for each base example
            else:
                each_alt_pred[row[1]] = row[-1] # Make a dict of predictions of alterations of this base example
    all_alt_pred.append(each_alt_pred)
    matched_pred = list(zip(base_pred, all_alt_pred)) # Zip each base and its alterations
    
    for a in alteration:
        nr_alt = 0
        nr_fail = 0
        for pair in matched_pred:
            if a in pair[1].keys(): # If this alteration exists for this base
                nr_alt += 1
                if pair[1][a] == pair[0]: # Fail if base and alteration have the same prediction
                    nr_fail += 1
                failure_rate[t][a] = nr_fail / nr_alt * 100

df = pd.DataFrame(failure_rate).T.astype(float).round(2).fillna('/').to_string()
print("Failure rate compared with base examples")
print("base: expected 'contradiction'; other: expected different")
print(df)

# Counting the distribution of predictions
neutral_count = 0
entail_count = 0
contra_count = 0
for row in content[1:]:
    if row[-1] == "neutral":
        neutral_count += 1
    elif row[-1] == "entailment":
        entail_count += 1
    elif row[-1] == "contradiction":
        contra_count += 1

# Counting the distribution of expected labels
exp_neu_count = 0
exp_ent_count = 0
exp_con_count = 0
for row in content[1:]:
    if row[3] == "neutral":
        exp_neu_count += 1
    if row[3] == "entailment":
        exp_ent_count += 1
    if row[3] == "contradiction":
        exp_con_count += 1

# Plotting the bar chart
labels = ['neutral', 'entailment', 'contradiction']
pred_count = [neutral_count, entail_count, contra_count]
exp_distri = [exp_neu_count, exp_ent_count, exp_con_count]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pred_count, width, label='Model predictions')
rects2 = ax.bar(x + width/2, exp_distri, width, label='Expected labels')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number')
ax.set_title('Distribution of model predictions and expected labels')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
