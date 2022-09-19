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

def calculate_conflicts(chessboard):
    '''make initial heuristic calculation'''
    conflicts = 0
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
                                conflicts += 1
                            '''if square is horizontal from chessboard[row][col]...initial_heuristic_cost+=1'''
                            if r == row and c != col:
                                conflicts += 1
    print("conflicts:", conflicts)
    return conflicts

'''ASSUMPTION: 1 queen per column'''
'''ASSUMPTION: only veritcal moves'''
def run(chessboard):
    global current_heuristic_cost
    print(current_heuristic_cost)
    for row in range(len(chessboard)):
        for col in range(len(chessboard[row])):
            '''for each non-queen square'''
            if chessboard[row][col][0] != "Q":
                print_board(chessboard)
                temp_chessboard = chessboard
                temp_chessboard[row][col] = "Q"
                for r in range(len(chessboard)):
                    for c in range(len(chessboard[r])):
                        if r == row:
                            if chessboard[r][c] == "Q":
                                temp_chessboard[r][c] = "Q"
                conflicts = calculate_conflicts(temp_chessboard)
                if conflicts == 0:
                    print("********YAY**********")
                    return chessboard
                if conflicts < initial_heuristic_cost:
                    chessboard[row][col][0] = "Q"
                    current_heuristic_cost = conflicts
                '''find the queen from the col and move it into chessboard[row][col]'''
                # for r in range(len(chessboard)):
                #     for c in range(len(chessboard[r])):
                #         if chessboard[r][c][0] == "Q" and c == col:
                #             conflicts = calculate_conflicts(chessboard)
                #             if conflicts == 0:
                #                 return chessboard
                #             if conflicts < initial_heuristic_cost:
                #                 chessboard[row][col] = chessboard[r][c]
                            # movement_cost = abs(r - row) * int(chessboard[r][c][1]) * int(chessboard[r][c][1])
                            # print(movement_cost)
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
    chessboard = run(chessboard)
    run_time = time.time() - start_time
    print_board(chessboard)
    print("total cost:", current_heuristic_cost, "run time:", run_time)