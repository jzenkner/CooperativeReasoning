import pandas as pd
import json
import ast
import numpy as np
import math

# Load CSV
df = pd.read_csv('./data/rules.csv')  # Replace with your path
df = df[df['run'] == 1]

NUM_EXAMPLES = 4


# Group by task id
grouped = df.groupby(['id'])

# Prepare data for JSONL
json_data = []
for idx, (task_id, group) in enumerate(grouped):
    inputs = []
    count = 0
    for inp in group['input1']:
        # Convert string like "[2, 4, 3, 2]" to list and then to desired format
        try:
            values = ast.literal_eval(inp)  # safe parsing of list
            formatted = f"x0 = [ {' '.join(str(x) for x in values)} ]"
        except Exception as e:
            raise ValueError(f"Error parsing input '{inp}': {e}")
        inputs.append(formatted)
        count += 1
        if count == NUM_EXAMPLES:
            count = 0
            break

    for idx, inp in enumerate(group['input2']):
        # Convert string like "[2, 4, 3, 2]" to list and then to desired format
        if math.isnan(inp) or inp == "None":
            continue
        try:
            values = int(inp)  # safe parsing of list
            formatted = f"x1 = {str(values)}"
        except Exception as e:
            raise ValueError(f"Error parsing input '{inp}': {e}")
        inputs[idx] += ' | ' + formatted
        count += 1
        if count == NUM_EXAMPLES:
            count = 0
            break

    outputs = []
    for out in group['output']:
        try:
            if out is np.nan or out == "None":
                formatted = "None"
            else:
                values = ast.literal_eval(out)
                # If values is not a list, treat it as a single value
                if not isinstance(values, list):
                    formatted = f"{values}"
                else:
                    formatted = f"[ {' '.join(str(x) for x in values)} ]"
        except Exception as e:
            raise ValueError(f"Error parsing input '{out}': {e}")
        outputs.append(formatted)
        count += 1
        if count == NUM_EXAMPLES:
            count = 0
            break

    program = group['concept'].iloc[0]

    assert len(inputs) == len(outputs)
    print({
        "index": idx,
        "inputs": inputs,
        "outputs": outputs,
        "program": program
    })
    json_data.append({
        "index": idx,
        "inputs": inputs,
        "outputs": outputs,
        "program": program
    })

# Write to JSONL file
with open('./data/test_data/deepcoder/RULES_ET_AL.jsonl', 'w') as f:
    for entry in json_data:
        f.write(json.dumps(entry) + '\n')