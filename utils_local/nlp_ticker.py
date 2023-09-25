import re

# Function to check if the string contains any numbers
def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))

# Function to check if the string contains both uppercase and lowercase letters
def has_mixed_case(input_string):
    return any(char.islower() for char in input_string) and any(char.isupper() for char in input_string)

def remove_lowercase(input_string):
    return ''.join(char for char in input_string if not char.islower())

def has_weird_symbols(input_string):
    return bool(re.search(r'[^a-zA-Z0-9]', input_string))

def remove_weird_symbols(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)

def clean_ticker(input_string,list_valid:list):
    invalid_output = ''
    # drop the one without . and remove markets
    if '.' not in input_string:
        return invalid_output
    else:
        input_string = input_string.split('.')[0]
    if len(input_string)==0:
        return invalid_output
    # remove the numbers ones
    if has_numbers(input_string):
        return invalid_output
    # remove the ipos
    for var in ['IPO-', 'IPO-']:
        if var in input_string:
            input_string = input_string.split(var)[0]
    # take care of lower letter representing shares numbers
    if has_mixed_case(input_string):
        input_string =remove_lowercase(input_string)
    # taking care of low caps after that
    input_string = input_string.upper()
    # remove remaining weird symbols
    input_string = remove_weird_symbols(input_string)
    if list_valid is not None:
        if input_string not in list_valid:
            return invalid_output, invalid_output
    return input_string



def get_market(input_string):
    try:
        return input_string.split('.')[1]
    except:
        return 'INVALID_MARKET'
