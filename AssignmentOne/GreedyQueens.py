import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings

#TODO: Implement Greedy (Willow)
#TODO: Add diagonal conflict resolution in run_Astar (Jesulona)
#TODO: Prove heuristic is admissible (Write-up)
#TODO: Part two is permitting queens to move horizontal in conjuntion to vertical (Kain)
#TODO: Final answer output needs exact sequence of movements to solution state (Spencer)

'''UNCOMMENT DESIRED CHESSBOARD'''
'''Note: The first chessboard is kinda too big to work on my computer'''
'''chessboard = np.array([[4, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 7],
                       [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 4, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])'''

'''chessboard = np.array([[4, 0, 0, 0, 9, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 9, 0]])'''

'''chessboard = np.array([[4, 2, 8, 8, 9, 3, 7, 3, 9, 2],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])'''

'''chessboard = np.array([[4, 0, 0, 0, 9, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0, 4, 0],
                       [0, 0, 0, 8, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 4],
                       [0, 0, 0, 0, 0, 0, 0]])'''

'''chessboard = np.array([[4, 0, 0, 0, 9],
                       [0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0],
                       [0, 0, 0, 8, 0],
                       [0, 0, 0, 0, 0]])'''

def count_conflict(chessboard):
    current_cost = 0
    Tchessboard = np.copy(chessboard)
    for i in range(0, len(Tchessboard)):
            for j in range(0, len(Tchessboard)):
                if (Tchessboard[j, i]) > 0:
                    if np.count_nonzero(Tchessboard[j]) >= 2:
                        #print(np.count_nonzero(Tchessboard[j]))
                        current_cost += np.count_nonzero(Tchessboard[j]) - 1
                    k = i - 1
                    l = j - 1
                    while k >= 0 and l >= 0:
                        if Tchessboard[l, k] > 0:
                            current_cost += 1
                        k += -1
                        l += -1
                    k = i - 1
                    l = j + 1
                    while k >= 0 and l < len(Tchessboard):
                        if Tchessboard[l, k] > 0:
                            current_cost += 1
                        k += -1
                        l += 1
                    k = i + 1
                    l = j + 1
                    while k < len(Tchessboard) and l < len(Tchessboard):
                        if Tchessboard[l, k] > 0:
                            current_cost += 1
                        k += 1
                        l += 1
                    k = i + 1
                    l = j - 1
                    while k < len(Tchessboard) and l >= 0:
                        if Tchessboard[l, k] > 0:
                            current_cost += 1
                        k += 1
                        l +=- 1
                Tchessboard[j, i] = 0
            #print(Tchessboard)
    #print(chessboard)
    return current_cost

def get_current_cost(chessboard):
    current_cost = 0
    for i in range(len(chessboard)):
        if np.count_nonzero(chessboard[i]) > 1:
            '''Squared weights to match squared movement cost. Most likely not Admissible.'''
            #current_cost = current_cost + np.sum(np.square(chessboard[i][chessboard[i] != 0]))
            '''Takes the squared weight of the minimum queen in conflict for each row.'''
            current_cost = current_cost + np.min(np.square(chessboard[i])[chessboard[i] != 0])
            '''Sum of weights of all queens in conflict, but doesn't match squared movement cost and isn't very good.'''
            #current_cost = current_cost + np.sum(chessboard[i])
            '''Cost calc for very small chessboards. Not good at all.'''
            #current_cost = current_cost + np.count_nonzero(chessboard[i])
    #return current_cost
    return count_conflict(chessboard) 


def run_Greedy(chessboard, fringe_chess, found_list):
    a = True
    while a == True:
        chosen_chess = np.argmin(fringe_chess[:, 1])
        #print("HEY LISTEN")
        print(found_list)
        for i in found_list:
            #print("AHHHH")
            if np.array_equal(fringe_chess[chosen_chess, 0],i):
                fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
                #print("QHY")
                continue
        print("HEY WHATSUP")
        found_list.append(fringe_chess[chosen_chess, 0])
        a = False
        break
    
    fringe_chess = np.array([[fringe_chess[chosen_chess,0], fringe_chess[chosen_chess,1], fringe_chess[chosen_chess,2], fringe_chess[chosen_chess,3], fringe_chess[chosen_chess,4]]])
    chosen_chess = 0
    for i in range(0, len(chessboard)):
        for j in range(0, len(chessboard)):
            if fringe_chess[chosen_chess, 0][j, i] > 0:
                for k in list(range(0,j)) + list(range(j+1,len(chessboard))):
                    temp_chess = np.copy(fringe_chess[chosen_chess, 0])
                    temp_chess[j, i], temp_chess[k, i] = temp_chess[k, i], temp_chess[j, i]
                    '''to get a greedy best-first approach is it as simple as cutting off one end of the below '+' symbol?'''
                    cost = get_current_cost(temp_chess)
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
    
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
        
        print(found_list)
        return fringe_chess[temp_index]
    
        '''If no solution, restarts the search with all opened nodes in the fringe_chess array.'''
    else:
        return run_Greedy(chessboard, fringe_chess, found_list)

'''Starts the search~'''
if __name__ == '__main__':
    '''select input csv'''
    inp_path = askopenfilename()
    print("user chose", inp_path)
    try:
        inp_csv = open(inp_path)
    except IOError as e:
        print(e)
        sys.exit()
    '''create chessboard'''
    df = pd.read_csv(inp_path, delimiter=',', header=None)
    chessboard = df.to_numpy()
    # chessboard = np.genfromtxt(inp_csv, delimiter=',', dtype='int32')
    '''convert nans to 0s'''
    chessboard[np.isnan(chessboard)] = 0
    print(repr(chessboard))
    '''suppress warnings'''
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    '''fringe_chess = chessboard, total cost, past cost, future cost'''
    fringe_chess = np.array([[chessboard, 0, 0, get_current_cost(chessboard), ""]])
    found_list = [0]
    '''start time'''
    start_time = time.time()
    print(run_Greedy(chessboard, fringe_chess, found_list))
    end_time = time.time()
    print("run-time:", end_time - start_time)

'''Notes for testing functions in case I need them'''
#print(np.any(chessboard[3, :] > 0))
#print(np.sum(chessboard[2]))
#chessboard[2, 1], chessboard[2, 2] = chessboard[2, 2], chessboard[2, 1]
#print(chessboard)
#print(np.count_nonzero(chessboard[1]))