import json
import os
os.chdir('../')
from tasks.lambdabeam.lambdabeam_dsl import Program

def parse_input_str(input_str):
    """
    Converts a string like 'x0 = 2 | x1 = [ 3 5 4 7 5 ]'
    into a dictionary of variable assignments.
    """
    parts = input_str.split('|')
    vars_dict = {}
    for p in parts:
        key, value = p.split('=')
        key = key.strip()
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            # Parse list
            value = [int(v) for v in value[1:-1].split() if v]
        else:
            value = int(value)
        vars_dict[key] = value
    return vars_dict

def load_tasks(jsonl_path):
    """Load all lines (tasks) from a JSONL file."""
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def truncate_inputs(inputs):
    """
    Truncate lists in the input to at most 5 elements.
    Handles both single-variable and two-variable inputs.

    Example:
        [[2, [3, 5, 4, 7, 5, 9]], [[1, 2, 3, 4, 5, 6]]]
        -> [[2, [3, 5, 4, 7, 5]], [[1, 2, 3, 4, 5]]]
    """
    truncated = []
    for item in inputs:
        # Each item can be like [2, [3, 5, 4, 7, 5]] or [[1, 2, 3, 4, 5, 6]]
        new_item = []
        for val in item:
            if isinstance(val, list):
                new_item.append(val[:5])
            else:
                new_item.append(val)
        truncated.append(new_item)
    return truncated

def format_value(val):
    """
    Formats a value for JSONL output:
    - Lists are formatted as '[ 1 2 3 4 ]' (no commas)
    - Integers remain as strings
    """
    if isinstance(val, (list, tuple)):
        return f"[ {' '.join(map(str, val))} ]"
    else:
        return str(val)


def main(jsonl_path):
    os.chdir('./TIIPS')
    with open("./data/test_data/lambdabeam/LAMBDABEAM_HANDWRITTEN.jsonl", 'w') as out_f:
      for task in load_tasks(jsonl_path):
          program_str = task['program2']
          if program_str == "x0 = INPUT | x1 = INPUT" or program_str == "x0 = INPUT":
              continue

          inputs = [parse_input_str(inp) for inp in task['inputs']]

          p = Program.from_str(program_str)

          inputs = [list(ele.values()) for ele in inputs]
          inputs = truncate_inputs(inputs)

          outputs = []
          for i, inp in enumerate(inputs):
              output = p.run(inp).get_output()
              for ii in inp:
                  if type(ii) == int and abs(ii) > 5:
                      print(str(p))


              outputs.append(str(output) if type(output) == int else "[ " + str(output)[1:-1].replace(',', '') + " ]")

          formatted = []
          for inp in inputs:
              if len(inp) == 2:
                  formatted.append(f"x0 = {format_value(inp[0])} | x1 = {format_value(inp[1])}")
              else:
                  formatted.append(f"x0 = {format_value(inp[0])}")

          new_task = {
            'index': task['index'],
            'inputs': formatted,
            'outputs': outputs,
            'program': task['program2']
          }
          out_f.write(json.dumps(new_task) + '\n')


if __name__ == "__main__":
    main("./data/test_data/lambdabeam/LAMBDABEAM_HANDWRITTEN_all.jsonl")
