"""
    Methods:
        run(chessboard)                                 main running function of hill climbing
        calculate_initial_conflict_scores(chessboard)   calculates initial heuristic
        calculate_conflicts(chessboard)                 calculates subsequent heuristics per square
        add_labels(chessboard)                          adds 'H' and 'Q' prefixes to queens and heuristic squares
        print_board(chessboard)                         helper function to print board in pretty way
    Global Variables:
        initial_heuristic_cost                          stores the starting heuristic cost of the board
        current_heuristic_cost                          stores the current cost so far
        num_queens                                      number of queens on the board
        moves                                           number of moves made to get solution
        is_complete                                     boolean to track completeness
        break_double_loop                               flag to know when to leave nested loops
        move_counter                                    tracks moves until restart
        reset_counter                                   count of number of restarts utilized
        MASTER_CHESSBOARD                               input chessboard, restart state
"""

import sys
import time
from tkinter.filedialog import askopenfilename
import csv
import copy

initial_heuristic_cost = 0
moves = 0
current_heuristic_cost = None
num_queens = 0
is_complete = False
break_double_loop = False
move_counter = 0
reset_counter = 0
MASTER_CHESSBOARD = None

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
                        '''if we are not comparing the same square, and the square is also a queen'''
                        if chessboard[r][c][0] == "Q":
                            '''if square is diagonal from chessboard[row][col]...initial_heuristic_cost+=1'''
                            '''if square is vertical from chessboard[row][col]...initial_heuristic_cost+=1'''
                            if c == col and r != row:
                                initial_heuristic_cost += 1
                            '''if square is horizontal from chessboard[row][col]...initial_heuristic_cost+=1'''
                            if r == row and c != col:
                                initial_heuristic_cost += 1
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
                        '''if we are not comparing the same square, and the square is also a queen'''
                        if chessboard[r][c][0] == "Q":
                            '''if square is diagonal from chessboard[row][col]...initial_heuristic_cost+=1'''
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
    global moves
    global is_complete
    global break_double_loop
    global move_counter
    global reset_counter
    while(not is_complete):
        for row in range(len(chessboard)):
            for col in range(len(chessboard[row])):
                print("move counter:", move_counter)
                if move_counter > 10:
                    chessboard = copy.deepcopy(MASTER_CHESSBOARD)
                    move_counter = 0
                    reset_counter += 1
                '''for each non-queen square'''
                if chessboard[row][col][0] != "Q":
                    '''try vertical moves first'''
                    temp_chessboard = copy.deepcopy(chessboard)
                    temp_chessboard[row][col] = "Q"
                    for r in range(len(chessboard)):
                        for c in range(len(chessboard[r])):
                            if c == col and r != row:
                                '''overwrite the original queen position with H0'''
                                if chessboard[r][c][0] == "Q":
                                    temp_chessboard[r][c] = "H0"
                                    break_double_loop = True
                                    break
                        if break_double_loop:
                            break
                        else:
                            continue
                    break_double_loop = False
                    conflicts = calculate_conflicts(temp_chessboard)
                    if conflicts == 0:
                        is_complete = True
                        print("TOOK A MOVE")
                        chessboard = copy.deepcopy(temp_chessboard)
                        moves += 1
                        move_counter += 1
                        print("********YAY**********")
                        return chessboard
                    if current_heuristic_cost is not None and conflicts < current_heuristic_cost:
                        chessboard = copy.deepcopy(temp_chessboard)
                        current_heuristic_cost = conflicts
                        print("TOOK A MOVE")
                        print_board(chessboard)
                        moves += 1
                        move_counter += 1
                        continue
                    elif current_heuristic_cost is None and conflicts < initial_heuristic_cost:
                        chessboard = copy.deepcopy(temp_chessboard)
                        print("TOOK A MOVE")
                        print_board(chessboard)
                        moves += 1
                        move_counter += 1
                        current_heuristic_cost = conflicts
                        continue
                    # '''try horizontal moves'''
                    # temp_chessboard = copy.deepcopy(chessboard)
                    # temp_chessboard[row][col] = "Q"
                    # for r in range(len(chessboard)):
                    #     for c in range(len(chessboard[r])):
                    #         if r == row and c != col:
                    #             if chessboard[r][c][0] == "Q":
                    #                 temp_chessboard[r][c] = "H0"
                    #                 break_double_loop = True
                    #                 break
                    #     if break_double_loop:
                    #         break
                    #     else:
                    #         continue
                    # break_double_loop = False
                    # conflicts = calculate_conflicts(temp_chessboard)
                    # if conflicts == 0:
                    #     is_complete = True
                    #     print("TOOK A HORIZONTAL MOVE")
                    #     chessboard = copy.deepcopy(temp_chessboard)
                    #     moves += 1
                    #     move_counter += 1
                    #     print("********YAY**********")
                    #     return chessboard
                    # if current_heuristic_cost is not None and conflicts < current_heuristic_cost:
                    #     chessboard = copy.deepcopy(temp_chessboard)
                    #     current_heuristic_cost = conflicts
                    #     print("TOOK A HORIZONTAL MOVE")
                    #     print_board(chessboard)
                    #     moves += 1
                    #     move_counter += 1
                    #     continue
                    # elif current_heuristic_cost is None and conflicts < initial_heuristic_cost:
                    #     chessboard = copy.deepcopy(temp_chessboard)
                    #     print("TOOK A HORIZONTAL MOVE")
                    #     print_board(chessboard)
                    #     moves += 1
                    #     move_counter += 1
                    #     current_heuristic_cost = conflicts
                    #     continue
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
    MASTER_CHESSBOARD = copy.deepcopy(chessboard)
    print_board(chessboard)
    '''start timer'''
    start_time = time.time()
    '''make first conflict calculation'''
    calculate_initial_conflict_scores(chessboard)
    chessboard = run(chessboard)
    run_time = time.time() - start_time
    print_board(chessboard)
    print("total moves:", moves, "run time:", run_time, "resets:", reset_counter)