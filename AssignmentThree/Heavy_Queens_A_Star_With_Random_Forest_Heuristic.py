from ast import Try
from sre_constants import SUCCESS
import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import matplotlib.pyplot as plt
import pickle

def conflict_stats(chessboard):
    count = 0
    sum = 0
    lic = 1000
    hic = 0
    avg = 0
    ldelta = 10000
    hdelta = 0
    cdelta = 0
    avgDelta = 0
    erows = 0
    mqueen_c = 0
    mqueen = 0
    lqueen_c = 15
    lqueen = 0
    avgqueens = 0
    nzrow = 0

    #for every space
    for row in range(len(chessboard)):
        for col in range(len(chessboard)):
            if chessboard[row][col] !=0: #IF YOU FIND A QUEEN
                #check row 
                for i in range(len(chessboard)):
                    if chessboard[row,i] !=0 and i!= col:
                        count +=1
                        sum += chessboard[row,i]
                        #distance
                        delta = abs(col-i)
                        cdelta += delta
                        if delta > hdelta:
                            hdelta = delta
                        if delta < ldelta:
                            ldelta = delta
                        #highest and lowest conflict
                        if chessboard[row,i] > hic:
                            hic = chessboard[row,i]
                        if chessboard[row,i] < lic and chessboard[row,i] != 0:
                            lic = chessboard[row,i]

                #check column 
                for j in range(len(chessboard)):
                    if chessboard[j,col] !=0 and j!= row:
                        count +=1
                        sum += chessboard[j,col]
                        #distance
                        delta = abs(row-j)
                        cdelta += delta
                        if delta > hdelta:
                            hdelta = delta
                        if delta < ldelta:
                            ldelta = delta
                        #highest and lowest conflict
                        if chessboard[j,col] > hic:
                            hic = chessboard[j,col]
                        if chessboard[j,col] < lic and chessboard[j,col] != 0:
                            lic = chessboard[j,col]
                
                #check L diag
                leftMost = find_upper_l(row,col,chessboard)
                rowMax = leftMost[0]
                colMax = leftMost[1]
                for x in range(0,len(chessboard)-max(rowMax,colMax)):
                    r= rowMax+x
                    c= colMax+x
                    if chessboard[r,c] != 0 and (r!=row and c!=col):
                        count +=1
                        sum += chessboard[r,c]
                        #distance
                        delta = math.sqrt(((row-r)**2 + (col-c)**2))
                        cdelta += delta
                        if delta > hdelta:
                            hdelta = delta
                        if delta < ldelta:
                            ldelta = delta
                        #highest and lowest conflict
                        if chessboard[r,c] > hic:
                            hic = chessboard[r,c]
                        if chessboard[r,c] < lic and chessboard[r,c] != 0:
                            lic = chessboard[r,c]

                #check R diag
                rightMost = find_upper_r(row,col,chessboard)
                rowMax = rightMost[0]
                colMax = rightMost[1]
                for x in range(0,colMax-rowMax+1):
                    r= rowMax+x
                    c= colMax-x
                    if chessboard[r,c] != 0 and (r!=row and c!=col):
                        count +=1
                        sum += chessboard[r,c]
                        #distance
                        delta = math.sqrt(((row-r)**2 + (col-c)**2))
                        cdelta += delta
                        if delta > hdelta:
                            hdelta = delta
                        if delta < ldelta:
                            ldelta = delta
                        #highest and lowest conflict
                        if chessboard[r,c] > hic:
                            hic = chessboard[r,c]
                        if chessboard[r,c] < lic and chessboard[r,c] != 0:
                            lic = chessboard[r,c]
        if np.sum(chessboard[row]) == 0:
            erows += 1
        if np.count_nonzero(chessboard[row]) > mqueen_c:
            mqueen_c = np.count_nonzero(chessboard[row])
            mqueen = row
        if np.count_nonzero(chessboard[row]) < lqueen_c:
            lqueen_c = np.count_nonzero(chessboard[row])
            lqueen = row
        if np.sum(chessboard[row]) != 0:
            avgqueens += np.count_nonzero(chessboard[row])
            nzrow += 1
    if count != 0:
        avg = sum/count
        avgDelta = cdelta/count
    if lic == 1000:
        lic = 0
    if ldelta == 10000:
        ldelta = 0
    avgqueens = avgqueens / nzrow
    return([lic, hic, avg, ldelta, hdelta, avgDelta, count])

def get_current_cost(chessboard, model):
    values = [] # df values
    # cols = [] # df col titles
    board = chessboard
    # attribute 1 - heaviest queen weight
    heaviestQueenWeight = np.amax(board)
    values.append(heaviestQueenWeight)
    # cols.append('heaviest queen weight')
    # attribute 2 - heaviest queen location
    values.append(unravel_index(board.argmax(), board.shape))
    # cols.append('location of heaviest queen')
    # attribute 3 - lightest queen
    i = np.unravel_index(np.where(board!=0, board, board.max()+1).argmin(), board.shape)
    lightestQueen = board[i]
    values.append(lightestQueen)
    # cols.append('lightest queen weight')
    # attribute 4 - location of lightest queen
    values.append(i)
    # cols.append('location of lightest queen')
    # attribute 5 - initial conflict
    initialConflict = spot_conflict(board)
    values.append(initialConflict)
    # cols.append('initial conflicts')
    # attribute 6 - n
    n = board.shape[0]
    values.append(n)
    # cols.append('n')
    # attribute 7 - average values including 0s
    avg = np.average(board)
    values.append(avg)
    # cols.append('average value including 0')
    #conflict statistics
    con_stats = conflict_stats(board)
    # attribute 8 - lightest queen in conflict
    lic = con_stats[0]
    values.append(lic)
    # cols.append('lightest in conflict')
    # attribute 9 - heaviest queen in conflict
    hic = con_stats[1]
    values.append(hic)
    # cols.append('heaviest in conflict')
    # attribute 10 - average weight of queens in conflict
    avgC = con_stats[2]
    values.append(avgC)
    # cols.append('average in conflict')
    # attribute 11 - smallest distance between conflict
    lDel = con_stats[3]
    values.append(lDel)
    # cols.append('smallest d in conflict')
    # attribute 12 - largest distance between conflict
    hDel = con_stats[4]
    values.append(hDel)
    # cols.append('largest d in conflict')
    # attribute 13 - average distance between conflicts
    avgD = con_stats[5]
    values.append(avgD)
    # cols.append('average d in conflict')
    # data to feed into model to get output
    return model.predict(values)

def run_Astar(chessboard, fringe_chess,found_list,start_time, model):
    model = model
    current_time = time.time() - start_time
    print(current_time)

    if current_time >= 30:
        return "Timed out", False

    a = True
    while a == True:
        chosen_chess = np.argmin(fringe_chess[:, 1])
        #print("HEY LISTEN")
        print(found_list)
        for i in found_list:
            #print("AHHHH")
            try:
                if np.array_equal(fringe_chess[chosen_chess, 0],i):
                    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
                    #print("QHY")
                    continue
            except IndexError:
                return "Index Error", False
        print("HEY WHATSUP")
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
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess, model), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
    
    '''Comment out if you only wanna print the final solution'''
    print(fringe_chess)
    
    '''Checks to see if a solution is found and returns the min solution if there is a solution.'''
    if 0 in fringe_chess[:, 3]:
        print("ANSWER:")
        answer = np.where(fringe_chess[:, 3] == 0)
        '''Initialized to an arbitrarily high number so the below if statement is true the first occasion, then future checks will go through'''
        temp_cost = 10000000000
        temp_index = -1
        for i in answer[0]:
            b = fringe_chess[i, 1]
            if fringe_chess[i, 1] < temp_cost:
                temp_index = i
                temp_cost = fringe_chess[i, 1]
        
        
        return fringe_chess[temp_index], True

    
        '''If no solution, restarts the search with all opened nodes in the fringe_chess array.'''
    else:
        return run_Astar(chessboard, fringe_chess,found_list,start_time)

def generate_random_board(n):
    '''This function will generate a random board of size n'''
    board = np.zeros((n, n))
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 9,n)
    return board

def experiment(n, model, niters=20,seed = 1):
    '''
    this function will run the experiment niters times of the n-queen problem
    '''
    model = model
    np.random.seed(seed)
    count = 0
    runtime = []
    chessboard = generate_random_board(n)
    fringe_chess = np.array([[chessboard, 0, 0, get_current_cost(chessboard), ""]])
    found_list = [0]
    start_time = time.time()

    for i in range(niters):
        board = generate_random_board(n)
        start_time = time.time()
        _, isConverged = run_Astar(board, fringe_chess,found_list,start_time, model)
        end_time = time.time()
        if isConverged:
            count+=1
            runtime.append(end_time-start_time)
    return runtime,count/niters


if __name__ == '__main__':
    with open('model.pkl','rb') as f:
        model = pickle.load(f)
    runtime, SUCCESS = experiment(5, model)
    print('runtime:', runtime, 'success rate:', SUCCESS)