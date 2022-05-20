from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import csv
import pandas as pd
from collections import defaultdict

# Loading the dataset
infile = "but_inferred_propositions.csv"
outfile = "./results/results.csv"

with open(infile, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    content = [row for row in reader] # Saving all the rows as `content'
    pairs = [[row[2], row[4]] for row in content[1:]] # Extracting only the premises and hypotheses

    
# Load model
tokenizer = T5Tokenizer.from_pretrained("t5-3b", force_download=True)
model = T5ForConditionalGeneration.from_pretrained("t5-3b", force_download=True)

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
perturbation = ['and', 'negation', 'temporal', 'sentiment', 'reversed']

failure_rate = defaultdict(dict)

for t in proposition_type:
    base_pred = []
    each_perturb_pred = {}
    all_perturb_pred = []
    for row in content[1:]: # Skip the header
        if row[0] == t:
            if row[1] == "":
                base_pred.append(row[-1]) # Make a list of 'base' predictions
                success = base_pred.count('contradiction')
                failure_rate[t][''] = (len(base_pred) - success) / len(base_pred) * 100
                if each_perturb_pred:
                    all_perturb_pred.append(each_perturb_pred) # Make a list of all predictions for all perturbations
                each_perturb_pred = {} # Reset the dict for each base example
            else:
                each_perturb_pred[row[1]] = row[-1] # Make a dict of predictions of perturbations of this base example
    all_perturb_pred.append(each_perturb_pred)
    matched_pred = list(zip(base_pred, all_perturb_pred)) # Zip each base and its perturbations
    
    for p in perturbation:
        nr_perturb = 0
        nr_fail = 0
        for pair in matched_pred:
            if p in pair[1].keys(): # If this perturbation exists for this base
                nr_perturb += 1
                if pair[1][p] == pair[0]: # Fail if base and perturbation have the same prediction
                    nr_fail += 1
                failure_rate[t][p] = nr_fail / nr_perturb * 100

# Printing the table
df = pd.DataFrame(failure_rate).T.astype(float).round(2).fillna('/').to_string()
print("Failure rate")
print("base: expected 'contradiction'; other: expected different")
print(df)
