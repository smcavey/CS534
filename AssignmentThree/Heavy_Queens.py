from ast import Try
from sre_constants import SUCCESS
import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import matplotlib.pyplot as plt

def find_upper_l(row,col,chessboard):
    #find uppermost lef diagonal
    while col >0 and row <0:
        row -=1
        col-=1
    return [row,col]

def find_upper_r(row,col,chessboard):
    #find uppermost right diagonal
    while row >0 and col < len(chessboard)-1:
        row -=1
        col+=1
    return [row,col]

def get_current_cost(chessboard):
    current_cost = 0
    for i in range(len(chessboard)):
        if np.count_nonzero(chessboard[i]) > 1:
            '''Squared weights to match squared movement cost. Most likely not Admissible.'''
            ###current_cost = current_cost + np.sum(np.square(chessboard[i][chessboard[i] != 0]))
            '''Takes the squared weight of the minimum queen in conflict for each row. Good heuristic'''
            current_cost = current_cost + np.min(np.square(chessboard[i])[chessboard[i] != 0])
            '''Sum of weights of all queens in conflict, but doesn't match squared movement cost and isn't very good.'''
            ###current_cost = current_cost + np.sum(chessboard[i])
            '''Cost calc for very small chessboards. Not good at all.'''
            ###current_cost = current_cost + np.count_nonzero(chessboard[i])
    '''chosen heuristic'''
    return current_cost 

def run_Astar(chessboard, fringe_chess,found_list,start_time):
    current_time = time.time() - start_time

    if current_time >= 30:
        return "Timed out", False

    a = True
    while a == True:
        chosen_chess = np.argmin(fringe_chess[:, 1])
        #print("HEY LISTEN")
        for i in found_list:
            #print("AHHHH")
            try:
                if np.array_equal(fringe_chess[chosen_chess, 0],i):
                    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
                    #print("QHY")
                    continue
            except IndexError:
                return "Index Error", False
        found_list.append(fringe_chess[chosen_chess, 0])
        a = False
        break

    for i in range(0, len(chessboard)):
        for j in range(0, len(chessboard)):
            if fringe_chess[chosen_chess, 0][j, i] > 0:
                #vertical movement
                for k in list(range(0,j)) + list(range(j+1,len(chessboard))):
                    temp_chess = np.copy(fringe_chess[chosen_chess, 0])
                    temp_chess[j, i], temp_chess[k, i] = temp_chess[k, i], temp_chess[j, i]
                    '''to get a greedy best-first approach is it as simple as cutting off one end of the below '+' symbol?'''
                    cost = abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + get_current_cost(temp_chess) + fringe_chess[chosen_chess, 2]
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
  
    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
    
    '''Checks to see if a solution is found and returns the min solution if there is a solution.'''
    if 0 in fringe_chess[:, 3]:
        answer = np.where(fringe_chess[:, 3] == 0)
        '''Initialized to an arbitrarily high number so the below if statement is true the first occasion, then future checks will go through'''
        temp_cost = 10000000000
        temp_index = -1
        for i in answer[0]:
            b = fringe_chess[i, 1]
            if fringe_chess[i, 1] < temp_cost:
                temp_index = i
                temp_cost = fringe_chess[i, 1]
        cost = fringe_chess[temp_index][1]
        return cost

    
        '''If no solution, restarts the search with all opened nodes in the fringe_chess array.'''
    else:
        return run_Astar(chessboard, fringe_chess,found_list,start_time)

def generate_random_board(n):
    '''This function will generate a random board of size n'''
    board = np.zeros((n, n))
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 10,n)
    return board

def experiment(n, df, niters=20,seed = 1):
    '''
    this function will run the experiment niters times of the n-queen problem
    '''
    np.random.seed(seed)
    count = 0
    runtime = []

    for i in range(niters):
        start_time = time.time()
        found_list = [0]
        board = generate_random_board(n)
        start_time = time.time()
        fringe_chess = np.array([[board, 0, 0, get_current_cost(board), ""]])
        cost = int(run_Astar(board, fringe_chess,found_list,start_time))
        heaviest_queen_weight = np.amax(board)
        i = np.unravel_index(np.where(board!=0, board, board.max()+1).argmin(), board.shape)
        lightest_queen_weight = board[i]
        flat = board.flatten()
        temp = pd.DataFrame([[n,cost, heaviest_queen_weight, lightest_queen_weight]], columns=['n','cost', 'heaviest', 'lightest'])
        df = df.append(temp, ignore_index=True)
    return df

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df = pd.DataFrame()
    df = experiment(5, df)
    df = experiment(6, df)
    df = experiment(7, df)
    df = experiment(8, df)
    df.to_csv('queen.csv')
