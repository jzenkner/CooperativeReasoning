import re
import json
import os


def is_single_string_to_string(sygus_code: str) -> bool:
    """
    Check if the SyGuS file defines a single string input -> string output function.
    """
    # Look for the synth-fun signature
    # Example: (synth-fun f ((_arg_0 String)) String)
    signature_pattern = r'\(synth-fun\s+f\s+\(\(\s*[A-Za-z_][A-Za-z0-9_]*\s+String\s*\)\)\s+String'
    return re.search(signature_pattern, sygus_code) is not None


def sygus_to_pbe_json(sygus_code: str):
    """
    Convert SyGuS PBE constraints into a JSON format with Examples.
    """
    # Match constraints of the form: (constraint (= (f "input") "output"))
    pattern = r'\(constraint\s*\(=\s*\(f((?:\s*"[^"]*")+)\)\s*(?:"([^"]*)"|(true|false))\s*\)\)'

    matches = re.findall(pattern, sygus_code, flags=re.IGNORECASE)

    examples = []
    for arg_block, str_out, bool_out in matches:
        # Extract all "..." arguments
        inputs = re.findall(r'"([^"]*)"', arg_block)
        # Determine output (string or boolean)
        output = str_out # if str_out else (bool_out.lower() == "true")
        examples.append({"Input": inputs, "Output": output})

    return {"Examples": examples}



if __name__ == "__main__":
    print(os.getcwd())
    input_dir = "data/sygus_slia_2019"
    output_dir = "data/test_data/robustfill"

    tasks = []
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith(".sl"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename.replace(".sl", ".json"))

            with open(in_path, "r", encoding="utf-8") as f:
                sygus_text = f.read()
            # print(filename.split('.')[0])
            pattern = r'\(constraint\s*\(=\s*\(f((?:\s*"[^"]*")+)\)\s*"([^"]*)"\s*\)\)'
            # r'\(constraint\s*\(=\s*\(f((?:\s*"[^"]*")+)\)\s*(?:"([^"]*)"|(true|false))\s*\)\)'

            matches = re.findall(pattern, sygus_text)

            # Construct task entry
            task = {
                "index": filename.split('.')[0],
                "inputs": [],  # keep quotes like in your format
                "outputs": [],
                "program": "unknown"  # placeholder — fill with actual DSL program later
            }

            c = 0
            for _, (arg_block, str_out) in enumerate(matches):
                # Extract input strings (keep empty ones)
                ipt = re.findall(r'"([^"]*)"', arg_block)
                # Skip if any input or output is missing (shouldn’t happen)
                if not ipt:
                    continue

                task['inputs'].append(str(ipt))
                task['outputs'].append(str(str_out))

                if c >= 3:
                    break
                c += 1
            if len(task['inputs']) == 0 or len(task['outputs']) == 0:
                continue

            tasks.append(task)
    print(len(tasks))
    """with open(output_dir + "/SYGUS_SLIA_2019_ALL.jsonl", "w", encoding="utf-8") as f_out:
        for t in tasks:
            f_out.write(json.dumps(t) + "\n")
    """