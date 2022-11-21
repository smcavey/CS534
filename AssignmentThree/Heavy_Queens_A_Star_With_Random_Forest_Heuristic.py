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
import os
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def find_upper_l(row,col,chessboard):
    #find uppermost lef diagonal
    while col >0 and row !=0:
        row -=1
        col-=1
    return [row,col]

def find_upper_r(row,col,chessboard):
    #find uppermost right diagonal
    while row !=0 and col < len(chessboard)-1:
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

def get_actual_conflicts(chessboard):
    current_cost = 0
    current_cost = spot_conflict(chessboard)
    return current_cost 

def conflict_stats(chessboard):
    count = 0
    sum = 0
    lic = 1000
    hic = 0
    avg = 0
    ldelta = 10000
    hdelta = 0
    cdelta = 0
    ldeltar = 0
    ldeltac = 0
    hdeltar = 0
    hdeltac = 0
    ldeltac = 0
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
                            hdeltac = delta
                        if delta < ldelta:
                            ldelta = delta
                            ldeltac = delta
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
                            hdeltar = delta
                        if delta < ldelta:
                            ldelta = delta
                            ldeltar = delta
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
                            hdeltar = row-r
                            hdeltac = col-c
                        if delta < ldelta:
                            ldelta = delta
                            ldeltar = row-r
                            ldeltac = col-c
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
                            hdeltar = row-r
                            hdeltac = col-c
                        if delta < ldelta:
                            ldelta = delta
                            ldeltar = row-r
                            ldeltac = col-c
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
    return([lic, hic, avg, ldelta, ldeltar, ldeltac, hdelta, hdeltar, hdeltac, avgDelta, count])

def get_current_cost(chessboard, model):
    values = [] # df values
    # cols = [] # df col titles
    board = chessboard
    # attribute 1 - heaviest queen weight
    heaviestQueenWeight = np.amax(board)
    values.append(heaviestQueenWeight)
    # cols.append('heaviest queen weight')
    # attribute 2 - heaviest queen row location
    values.append(np.unravel_index(board.argmax(), board.shape)[0])
    # cols.append('row location of heaviest queen')
    # attribute 3 - heaviest queen col location
    values.append(np.unravel_index(board.argmax(), board.shape)[1])
    # cols.append('col location of heaviest queen')
    # attribute 4 - lightest queen
    i = np.unravel_index(np.where(board!=0, board, board.max()+1).argmin(), board.shape)
    lightestQueen = board[i]
    values.append(lightestQueen)
    # cols.append('lightest queen weight')
    # attribute 5 - lightest queen row location
    values.append(i[0])
    # cols.append('row location of lightest queen')
    # attribute 6 - lightest queen col location
    values.append(i[1])
    # cols.append('col location of lightest queen')
    # attribute 7 - initial conflict
    initialConflict = conflict_stats(board)[10]
    values.append(initialConflict)
    # cols.append('initial conflicts')
    # attribute 8 - n
    n = board.shape[0]
    values.append(n)
    # cols.append('n')
    # attribute 9 - average values including 0s
    avg = np.average(board)
    values.append(avg)
    # cols.append('average value including 0')
    #conflict statistics
    con_stats = conflict_stats(board)
    # attribute 10 - lightest queen in conflict
    lic = con_stats[0]
    values.append(lic)
    # cols.append('lightest in conflict')
    # attribute 11 - heaviest queen in conflict
    hic = con_stats[1]
    values.append(hic)
    # cols.append('heaviest in conflict')
    # attribute 12 - average weight of queens in conflict
    avgC = con_stats[2]
    values.append(avgC)
    # cols.append('average in conflict')
    # attribute 13 - smallest distance between conflict- vector
    lDel = con_stats[3]
    values.append(lDel)
    # cols.append('smallest d in conflict')
    # attribute 14 - smallest row distance between conflict
    lDelr = con_stats[4]
    values.append(lDelr)
    # cols.append('smallest row d in conflict')
    # attribute 15 - smallest col distance between conflict
    lDelc = con_stats[5]
    values.append(lDelc)
    # cols.append('smallest d in conflict')
    # attribute 16 - largest distance between conflict
    hDel = con_stats[6]
    values.append(hDel)
    # cols.append('largest d in conflict')
    # attribute 17 - largest row distance between conflict
    hDelr = con_stats[7]
    values.append(hDelr)
    # cols.append('largest row d in conflict')
    # attribute 18 - largest col distance between conflict
    hDelc = con_stats[8]
    values.append(hDelc)
    # cols.append('largest d in conflict')
    # attribute 19 - average distance between conflicts
    avgD = con_stats[9]
    values.append(avgD)

    # temp = convert_float(values[1])
    # values[1] = temp
    # temp = convert_float(values[3])
    # values[3] = temp
    values = np.array(values)
    values = values.reshape(1, -1)
    # cols.append('average d in conflict')
    # data to feed into model to get output
    cost = int(model.predict( values ))
    # print('cost', cost)
    return cost

def run_Astar(chessboard, fringe_chess,found_list,start_time, model):
    model = model
    current_time = time.time() - start_time
    #print('current time', current_time)

    if current_time >= 180:
        return "Timed out", False, -1

    a = True
    while a == True:
        chosen_chess = np.argmin(fringe_chess[:, 1])
        #print("HEY LISTEN")
        #print(found_list)
        for i in found_list:
            #print("AHHHH")
            try:
                if np.array_equal(fringe_chess[chosen_chess, 0],i):
                    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
                    #print("QHY")
                    continue
            except IndexError:
                return "Index Error", False, -1
        #print("HEY WHATSUP")
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
                    cost = abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + get_current_cost(temp_chess, model) + fringe_chess[chosen_chess, 2]
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess, model), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
    
    '''Comment out if you only wanna print the final solution'''
    #print(fringe_chess)
    
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
        fin_conflicts = get_actual_conflicts(fringe_chess[chosen_chess][0])
        print(fringe_chess[chosen_chess][0])
        return fringe_chess[temp_index][0], True, fin_conflicts

    
        '''If no solution, restarts the search with all opened nodes in the fringe_chess array.'''
    else:
        return run_Astar(chessboard, fringe_chess,found_list,start_time, model)

def generate_random_board(n):
    '''This function will generate a random board of size n'''
    board = np.zeros((n, n))
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 9,n)
    return board

def experiment(n, model, niters,seed = 1):
    '''
    this function will run the experiment niters times of the n-queen problem
    '''
    model = model
    np.random.seed(seed)
    count = 0
    runtime = []

    found_list = [0]
    for i in range(niters):
        print("RUN ", i + 1, ":")
        chessboard = generate_random_board(n)
        print("INITIAL BOARD: ")
        print(chessboard)
        fringe_chess = np.array([[chessboard, 0, 0, get_current_cost(chessboard,model), ""]])
        begin_con = get_actual_conflicts(chessboard)
        start_time = time.time()
        isConverged = None
        start_time = time.time()
        _, isConverged, fin_con = run_Astar(chessboard, fringe_chess,found_list,start_time, model)
        end_time = time.time()
        try:
            if fin_con > -1:
                print("SOLUTION QUALITY: ", str(abs(1 - (float(fin_con/begin_con)))))
        except ZeroDivisionError:
            if fin_con != 0:
                print("SOLUTION QUALITY: Introduced conflict on a solved board 0")
            else:
                print("SOLUTION QUALITY: 0")
        print("RUN TIME: ", str(end_time-start_time))
        if isConverged:
            count+=1
            runtime.append(end_time-start_time)
    return runtime,count/niters

def convert_float(inp):
    inp = str(inp)
    inp = inp.replace("(", "")
    inp = inp.replace(")", "")
    splitted_data = inp.split(",")
    temp = str(splitted_data[0] + '.' + splitted_data[1])
    temp = temp.replace(" ", "")
    return temp

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    dir = os.path.dirname(os.path.abspath(__file__))
    attr = 'attributes.csv'
    attrPath = os.path.join(dir, attr)
    n = int(input("Enter board size n: "))
    iter = int(input("Enter number of boards to run: "))
    # with open(modelPath,'rb') as f:
    # model = pickle.load(f)
    data = pd.read_csv(attrPath)
    # train the model
    Y = data['cost']
    X = data.loc[:, data.columns != 'cost']
    model = RandomForestClassifier(max_depth=20)
    X = X.values
    # scores = cross_val_score(model, X, Y, cv=10)
    print("TRAINING...")
    model.fit(X, Y)
    runtime, SUCCESS = experiment(n, model, iter)
    print('SUCCESS RATE: ', SUCCESS)
    #print('model 10x accuracy scores', scores)