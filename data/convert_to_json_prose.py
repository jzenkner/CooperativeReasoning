import os
import json

input_root = "data/prose_text_trafo"
output_file = "data/test_data/robustfill/PROSE_TEXT_TRAFO.jsonl"

task_index = 0

with open(output_file, "w", encoding="utf-8") as f_out:
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file != "spec.json":
                continue

            spec_path = os.path.join(root, file)
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)

            examples = spec.get("Examples", [])
            if not examples:
                continue
            
            # Collect inputs and outputs separately
            inputs_list = []
            outputs_list = []
            c = 0
            for ex in examples:
                if len(ex['Input']) > 1 or not ex['Output']:
                    continue
                inputs = ex['Input']
                output = ex['Output']
                inputs_list.extend(inputs)
                outputs_list.append(output)

                if c >= 3:
                    break
                c += 1
            
            if c < 3:
                continue
            
            # Construct task
            task = {
                "index": root.split('/')[-1],
                "inputs": inputs_list,
                "outputs": outputs_list,
                "program": ""
            }

            if not (len(inputs_list) == len(outputs_list) == 4):
                print(task)
                print(len(inputs_list), len(outputs_list))
                continue

            f_out.write(json.dumps(task) + "\n")
            task_index += 1

print(f"Saved {task_index} tasks to {output_file}")
