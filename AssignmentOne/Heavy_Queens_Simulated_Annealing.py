"""
    Methods:
        run(chessboard)                                 main running function of simulated annealing
        find_upper_l(row,col,chessboard)                finds leftmost cell relative to current
        find_upper_r(row,col,chessboard)                finds rightmost cell relative to current
        spot_conflict(chessboard)                       calculates num conflict (will double count)
        make_random_move(chessboard, queen)             moves a queen up or down by one
        sample(chessboard)                              picks a queen to move
        print_board(chessboard)                         helper function to print board in pretty way
    Global Variables:
        
"""

from distutils.log import info
import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import csv
import random
import math
import copy
import matplotlib.pyplot as plt



def find_upper_l(row,col,chessboard):
    #find uppermost lef diagonal
    while col > 0 and row < 0:
        row -=1
        col-=1
    return [row,col]

def find_upper_r(row,col,chessboard):
    #find uppermost right diagonal
    while row > 0 and col < len(chessboard)-1:
        row -=1
        col+=1
    return [row,col]

def spot_conflict(chessboard):
    count = 0
    #for every space
    for row in range(len(chessboard)):
        for col in range(len(chessboard)):
            if chessboard[row,col] !=0: #IF YOU FIND A QUEEN
                #check row 
                for i in range(len(chessboard)):
                    if chessboard[row,i] !=0 and i!= col:
                        count +=1
                #check column 
                for j in range(len(chessboard)):
                    if chessboard[j,col] !=0 and j!= row:
                        count +=1

                #check L diag
                leftMost = find_upper_l(row,col,chessboard)
                rowMax = leftMost[0]
                colMax = leftMost[1]
                for x in range(0,len(chessboard)-max(rowMax,colMax)):
                    r= rowMax+x
                    c= colMax+x
                    if chessboard[r,c] != 0 and (r!=row and c!=col):
                        count +=1

                #check R diag
                rightMost = find_upper_r(row,col,chessboard)
                rowMax = rightMost[0]
                colMax = rightMost[1]
                for x in range(0,colMax-rowMax+1):
                    r= rowMax+x
                    c= colMax-x
                    if chessboard[r,c] != 0 and (r!=row and c!=col):
                        count +=1
    return(count)

def make_random_move(chessboard, queen):
    row = queen[0]
    col = queen[1]
    val = queen[2]

    nrow = 0

    if row == len(chessboard)-1: #if at bottom
        nrow = row-1
        chessboard[nrow,col] = val # move up
        chessboard[row,col] = 0
    elif row == 0: #if at top
        nrow = row+1
        chessboard[nrow,col] = val # move down
        chessboard[row,col] = 0
    else:
        MOE = random.choice([1,-1]) #move up or down by one
        nrow = row+MOE
        chessboard[nrow,col] = val # magic occurs here
        chessboard[row,col] = 0
    return [chessboard, "Move {},{} to {},{}. ".format(row, col, nrow, col)]
    

def sample(chessboard):
    queens = []
    data =[]

    #for every space
    for row in range(len(chessboard)):
        for col in range(len(chessboard)):
            if chessboard[row,col] !=0: #IF YOU FIND A QUEEN
                qInfo = [row,col,chessboard[row][col]] #row,col,weight
                queens.append(qInfo)
    queen = random.choice(queens) #pick one
    
    moveCost = queen[2]**2 
    move = make_random_move(chessboard,queen)
    newChessboard = move[0]
    moveText = move[1]
    heuristic = spot_conflict(newChessboard)

    data.append(moveCost)
    data.append(heuristic)
    data.append(newChessboard)
    data.append(moveText)
    return data
    

'''ASSUMPTION: only single veritcal moves: see make_random_move()'''
def run(chessboard,start_time):
    current_time = time.time() - start_time
    print(current_time)

    if current_time >= 30:
        return "Timed out", False

    temp = 10000000 #will always solve with v high temp for 7X7. Lower temp ok for smaller board
    #decay = 100
    decay = 0.95
    currentFit = 0 + spot_conflict(chessboard)
    
    moves = list()
    #total_cost = 0
   
    while temp > 0 and spot_conflict(chessboard)!=0:
        boardSample = sample(chessboard)

        moveCost = boardSample[0]
        heuristic = boardSample[1]
        newChessboard = boardSample[2]
        moveText = boardSample[3]

        '''Can vary fit to include cost to move or not'''
        newFit = moveCost + heuristic
        #newFit = heuristic

        if newFit <= currentFit:
            currentFit = newFit
            chessboard = newChessboard
            moves.append(moveText)
        else :
            delta = -(newFit-currentFit) #was unsure abt - but it seems to work better with - i think its cause of the way < heuristic is better
            k = 1
            power = delta/(k*temp)
           
            prob = math.exp(power)
            diceroll = random.uniform(0, 1)
            if diceroll <= prob:
                currentFit = newFit
                chessboard = newChessboard
                moves.append(moveText)


        #temp -=decay #decrease temperature
        temp = temp/(1+decay*temp)
    
    return [chessboard, moves], True


def print_board(chessboard):
    for row in range(len(chessboard)):
        print(chessboard[row])

def generate_random_board(n):
    '''This function will generate a random board of size n'''
    board = np.zeros((n, n))
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 9,n)
    return board

def experiment(n, niters=20,seed = 1):
    '''
    this function will run the experiment niters times of the n-queen problem
    '''
    np.random.seed(seed)
    count = 0
    runtime = []
    start_time = time.time()

    for i in range(niters):
        board = generate_random_board(n)
        start_time = time.time()
        _, isConverged = run(board,start_time)
        end_time = time.time()
        if isConverged:
            count+=1
            runtime.append(end_time-start_time)
    return runtime,count/niters


# if __name__ == '__main__':
#     runtime, SUCCESS = experiment(5)
#     #ploting
#     #you need the array for n4-n17.
#     boxplot = plt.violinplot([runtime],showmeans=True)
#     #boxplot = df.boxplot(column=['4', '5', '6','7','8','9'],figsize = (12,12))
#     plt.yscale('log')
#     plt.ylim(0,1000)    
#     plt.xlabel("n")
#     plt.ylabel("time (s)")
#     plt.title("Time used for solving n-queen problem with A* and horizontal moves enabled")
#     plt.show()


if __name__ == '__main__':
    runtime, SUCCESS = experiment(8)
    #ploting
    #you need the array for n4-n17.
    boxplot = plt.violinplot([runtime],showmeans=True)
    #boxplot = df.boxplot(column=['4', '5', '6','7','8','9'],figsize = (12,12))
    plt.yscale('log')
    plt.ylim(0,1000)    
    plt.xlabel("n")
    plt.ylabel("time (s)")
    plt.title("Time used for solving n-queen problem with Simulated Annealing")
    plt.show()

# '''Starts the search~'''
# if __name__ == '__main__':
#     '''select input csv'''
#     inp_path = askopenfilename()
#     print("user chose", inp_path)
#     try:
#         inp_csv = open(inp_path)
#     except IOError as e:
#         print(e)
#         sys.exit()
#     '''create chessboard'''
#     df = pd.read_csv(inp_path, delimiter=',', header=None)
#     chessboard = df.to_numpy()
#     '''convert nans to 0s'''
#     chessboard[np.isnan(chessboard)] = 0
#     '''suppress warnings'''
#     warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#     '''start time'''
#     start_time = time.time()
#     result,_ = run(chessboard, start_time)
#     end_time = time.time()
#     print_board(result[0])
#     print("run-time:", end_time - start_time)
#     print("moves: ", result[1])
#     if spot_conflict(result[0]) == 0:
#         print("YAY")