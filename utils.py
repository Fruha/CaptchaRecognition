import os
import json
import torch

def save_in_json(data: dict, path: str, indent: int = 4) -> None:
    """
    Save data in Json object
    Arguments:
        data: dict
        path: path to save file
        indent = 4: count of spaces for indent
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def load_from_json(path: str) -> dict:
    """
    Load data from json file
    Arguments:
        path: path to file
    Return:
        data from file\n
        None if file don't exists
    """
    if os.path.exists(path):
        with open(path) as json_file:
            data = json.load(json_file)
            return data
    else:
        return {}

def decode_str(data: list, char_set_pred: str) -> str:
    """
    Decode list to str using char_set_pred as decoder
    Arguments:
        data: list of ints
        char_set_pred: set of chars for decode
    Return:
        Decoded str
    """
    decoder = {key: val for key, val in enumerate(char_set_pred)}
    data = ''.join([decoder[i] for i in data])
    return data

def ctc_decoder(data: torch.Tensor, char_set_pred: str = None):
    """
    Decode CTC output
    Arguments:
        data: (L,C)
        char_set_pred: str of all chars
    Return:
        Tensor if char_set_pred is None\n
        Str if char_set_pred is not None
    """
    data = data.argmax(1)
    data = torch.unique_consecutive(data)
    data = data[data != 0]
    if char_set_pred is not None:
        data = decode_str(data.cpu().numpy(), char_set_pred)
    return data