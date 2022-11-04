from ast import Try
from sre_constants import SUCCESS
import numpy as np
from numpy import genfromtxt
import ast
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

def spot_conflict(chessboard):
    count = 0
    #for every space
    for row in range(len(chessboard)):
        for col in range(len(chessboard)):
            if chessboard[row][col] !=0: #IF YOU FIND A QUEEN
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

def get_current_cost(chessboard):
    return spot_conflict(chessboard) 

def run_Astar(chessboard, fringe_chess,found_list,start_time):
    current_time = time.time() - start_time

    if current_time >= 30:
        return -1

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
                return -1
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
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 9,n)
    return board

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    warnings.simplefilter(action='ignore', category=FutureWarning)
    inp = pd.read_csv('test.csv')
    out = pd.DataFrame()
    for index, row in inp.iterrows(): # do work
        values = [] # df values
        cols = [] # df col titles
        board = row['board']
        #board = board.replace('.', ',')
        new = None
        for row in board:
            temp = row + ','
            new = ''.join(temp)
        board = new
        board = genfromtxt(board, delimiter=',')
        print(board)
        cost = row['cost'] # target variable
        values.append(cost)
        cols.append('cost')
        heaviestQueenWeight = np.amax(board)
        values.append(heaviestQueenWeight)
        cols.append('heaviest queen weight')
        temp = pd.DataFrame([values], columns=cols)
        out = out.append(temp, ignore_index=True)
    out.to_csv('attributes.csv')
