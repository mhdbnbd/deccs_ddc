import os
import argparse
import json

def str2bool(v):
    """
    Argparse utility function for extracting boolean arguments.
    Original code is from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        print("returned true")
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2device(v):
    """
    Argparse utility function for checking that input is either cpu or cuda (gpu)
    """
    if v.lower() in ("cpu"):
        return "cpu"
    elif "cuda" in v.lower():
        return v.lower()
    else:
        raise argparse.ArgumentTypeError('cuda or cpu expected, but got {}.'.format(str(v)))



def json2dict(file_path):
    """
    Argparse utility function for converting json to dictionary.
    """
    if file_path is not None:
        try:
            dictionary = json.load(open(file_path, "r"))
        except Exception as e:
            print('Json value expected: ', e)
        return dictionary
