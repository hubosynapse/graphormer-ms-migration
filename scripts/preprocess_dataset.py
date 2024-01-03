import json
from mindnlp.transformers import preprocess_item


def jsonl_file_iterator(file_path, column_names):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            jline = json.loads(line)
            values = []
            for cn in column_names:
                if cn not in jline:
                    ValueError(f"Key {cn} not contained in {line} in the input json file")
                values.append(jline[cn])

            yield values

def collate_function():
    pass


def process_jsonl(input_file_path, output_file_path, collate_function):
    """
    Process a .jsonl file: read each line, apply collate_function, and write to a new file.

    :param input_file_path: Path to the input .jsonl file
    :param output_file_path: Path to the output .jsonl file
    :param collate_function: Function to apply to each line
    """
    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                # Parse JSON from each line
                json_obj = json.loads(line.strip())

                # Apply the collate function
                new_json_obj = collate_function(json_obj)

                # Write the transformed object to the output file
                output_file.write(json.dumps(new_json_obj) + '\n')

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    data_dir = "../../graphormer-data/ogbg-molhiv/"
    pattern = os.path.join(directory_path, "*.jsonl")
    jsonl_files = glob.glob(pattern)

    for jfile in jsonl_files:
        cfile = re.replace(r"\.jsonl$", "_collated.jsonl", jfile)
        process_jsonl(jfile, cfile, collate_function)
