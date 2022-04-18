from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import csv

# Loading the dataset
infile = "but_inferred_propositions.csv"
outfile = "./results/results.csv"

with open(infile, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    content = [row for row in reader] # Saving all the rows as `content'
    pairs = [[row[2], row[4]] for row in content[1:]] # Extracting only the premises and hypotheses

# Load model
tokenizer = T5Tokenizer.from_pretrained("t5-small") # t5-11b
model = T5ForConditionalGeneration.from_pretrained("t5-small") # t5-11b

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

