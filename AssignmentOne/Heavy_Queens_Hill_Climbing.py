"""
Methods:
    calculate_conflict_scores(chessboard)           calculates cheapest value for all squares
    add_labels(chessboard)                          adds 'H' and 'Q' prefixes to queens and heuristic squares
    print_board(chessboard)                         helper function to print board in pretty way
"""

import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import csv

def calculate_conflict_scores(chessboard):
    return chessboard

def add_labels(chessboard):
    for row in range(len(chessboard)):
        for col in range(len(chessboard[row])):
            if chessboard[row][col] == "0":
                chessboard[row][col] = "H" + chessboard[row][col]
            else:
                chessboard[row][col] = "Q" + chessboard[row][col]
    return chessboard

def print_board(chessboard):
    for row in range(len(chessboard)):
        print(chessboard[row])

if __name__ == '__main__':
    '''select input csv'''
    inp_path = askopenfilename()
    '''read in csv as 2d list'''
    try:
        chessboard = list(csv.reader(open(inp_path)))
    except IOError as e:
        sys.exit(e)
    start_time = time.time()
    '''add string formatting to differentiate queens/weights from heuristic'''
    chessboard = add_labels(chessboard)
    print_board(chessboard)
    '''make first conflict calculation'''
    # chessboard = calculate_conflict_scores(chessboard)