def parse_record(record):
    """
    record (str): string formatted as a dictionary. each new line is a key-value pair
    Return:
        record_dict: dictionary extracted from the string
    """
    record_dict = dict()
    pairs = record.split("\n")[1:-1] # exclude parenthesis
    for i, pair in enumerate(pairs):
        pair = list(map(lambda x: x.strip('\'\'\"\" ,'), pair.split(': ')))
        key = pair[0]

        
        if "}" in key:
            return record_dict
        elif key in ["impact", "population", "infrastructures"]:
            value = parse_record("\n".join(pairs[i:]))
            record_dict[key] = value
            break
        value = pair[1]
        record_dict[key] = value

    return record_dict

def get_records_string(file):
    opening = "{"
    open_parenthesis = 1
    record = opening

    for line in file:
        record += line

        # FIFO queue to handle parenthesis
        if "}" in line:
            open_parenthesis -= 1
            if open_parenthesis == 0:
                break
        elif "{" in line:
            open_parenthesis += 1

    return record