"""
    Methods:
        run(chessboard)                                 main running function of hill climbing
        calculate_initial_conflict_scores(chessboard)   calculates initial heuristic
        calculate_followup_conflict_scores(chessboard)  calculates subsequent heuristics per square
        add_labels(chessboard)                          adds 'H' and 'Q' prefixes to queens and heuristic squares
        print_board(chessboard)                         helper function to print board in pretty way
    Global Variables:
        initial_heuristic_cost                          stores the starting heuristic cost of the board
        current_heuristic_cost                          stores the current cost so far
        num_queens                                      number of queens on the board
"""

import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import csv

initial_heuristic_cost = 0
current_heuristic_cost = None
num_queens = 0

def run(chessboard):
    pass

def calculate_initial_conflict_scores(chessboard):
    '''make initial heuristic calculation'''
    global initial_heuristic_cost
    for row in range(len(chessboard)):
        '''for each square on the board'''
        for col in range(len(chessboard[row])):
            '''if the square has a queen on it'''
            if chessboard[row][col][0] == "Q":
                '''check all vertical, diagonal, and horizontal queens'''
                for r in range(len(chessboard)):
                    for c in range(len(chessboard[r])):
                        # print("row:", row, "col:", col, "r:", r, "c:", c)
                        '''if we are not comparing the same square, and the square is also a queen'''
                        if chessboard[r][c][0] == "Q":
                            '''if square is diagonal from chessboard[row][col]...initial_heuristic_cost+=1'''
                            # TODO: increment initial_heuristic_cost for each diagonal queen
                            '''if square is vertical from chessboard[row][col]...initial_heuristic_cost+=1'''
                            if c == col and r != row:
                                initial_heuristic_cost += 1
                            '''if square is horizontal from chessboard[row][col]...initial_heuristic_cost+=1'''
                            if r == row and c != col:
                                initial_heuristic_cost += 1
    print("initial board heuristic:", initial_heuristic_cost)
    return None

'''ASSUMPTION: 1 queen per column'''
'''ASSUMPTION: only veritcal moves'''
def calculate_followup_conflict_scores(chessboard):
    temp_chessboard = chessboard
    for row in range(len(chessboard)):
        for col in range(len(chessboard[row])):
            '''for each non-queen square'''
            if chessboard[row][col][0] != "Q":
                pass
                #TODO: assume each queen in each colummn moves to each other square, calculate conflicts, put best in each square
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
    num_queens = len(chessboard)
    '''add string formatting to differentiate queens/weights from heuristic'''
    chessboard = add_labels(chessboard)
    print_board(chessboard)
    '''start timer'''
    start_time = time.time()
    '''make first conflict calculation'''
    calculate_initial_conflict_scores(chessboard)
    #run(chessboard)
    run_time = time.time() - start_time
    chessboard = calculate_followup_conflict_scores(chessboard)
    print_board(chessboard)
    print("total cost:", current_heuristic_cost, "run time:", run_time)