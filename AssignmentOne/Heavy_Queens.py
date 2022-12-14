from ast import Try
from sre_constants import SUCCESS
import numpy as np
import sys
import time
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
import matplotlib.pyplot as plt

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
    current_cost = 0
    for i in range(len(chessboard)):
        if np.count_nonzero(chessboard[i]) > 1:
            '''Squared weights to match squared movement cost. Most likely not Admissible.'''
            ###current_cost = current_cost + np.sum(np.square(chessboard[i][chessboard[i] != 0]))
            '''Takes the squared weight of the minimum queen in conflict for each row. Good heuristic'''
            #current_cost = current_cost + np.min(np.square(chessboard[i])[chessboard[i] != 0])
            '''Sum of weights of all queens in conflict, but doesn't match squared movement cost and isn't very good.'''
            ###current_cost = current_cost + np.sum(chessboard[i])
            '''Cost calc for very small chessboards. Not good at all.'''
            ###current_cost = current_cost + np.count_nonzero(chessboard[i])
    '''chosen heuristic'''
    current_cost = spot_conflict(chessboard)
    return current_cost 

def run_Astar(chessboard, fringe_chess,found_list,start_time):
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
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
                # # #horizontal movement
                for l in list(range(0,i)) + list(range(i+1,len(chessboard))):
                    temp_chess = np.copy(fringe_chess[chosen_chess, 0])
                    temp_chess[j, i], temp_chess[j, l] = temp_chess[j, l], temp_chess[j, i]
                    cost = abs(i - l) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + get_current_cost(temp_chess) + fringe_chess[chosen_chess, 2]
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(i - l) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, j, l)]]), axis=0)
  
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

def experiment(n, niters=20,seed = 1):
    '''
    this function will run the experiment niters times of the n-queen problem
    '''
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
        _, isConverged = run_Astar(board, fringe_chess,found_list,start_time)
        end_time = time.time()
        if isConverged:
            count+=1
            runtime.append(end_time-start_time)
    return runtime,count/niters


if __name__ == '__main__':
    runtime, SUCCESS = experiment(5)
    #ploting
    #you need the array for n4-n17.
    boxplot = plt.violinplot([runtime],showmeans=True)
    #boxplot = df.boxplot(column=['4', '5', '6','7','8','9'],figsize = (12,12))
    plt.yscale('log')
    plt.ylim(0,1000)    
    plt.xlabel("n")
    plt.ylabel("time (s)")
    plt.title("Time used for solving n-queen problem with A* and horizontal moves enabled")

'''Starts the search~'''
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
#     # chessboard = np.genfromtxt(inp_csv, delimiter=',', dtype='int32')
#     '''convert nans to 0s'''
#     chessboard[np.isnan(chessboard)] = 0
#     print(repr(chessboard))
#     '''suppress warnings'''
#     warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#     '''fringe_chess = chessboard, total cost, past cost, future cost'''
#     fringe_chess = np.array([[chessboard, 0, 0, get_current_cost(chessboard), ""]])
#     found_list = [0]
#     '''start time'''
#     start_time = time.time()
#     print(run_Astar(chessboard, fringe_chess,found_list,start_time))
#     end_time = time.time()
#     print("run-time:", end_time - start_time)

'''Notes for testing functions in case I need them'''
#print(np.any(chessboard[3, :] > 0))
#print(np.sum(chessboard[2]))
#chessboard[2, 1], chessboard[2, 2] = chessboard[2, 2], chessboard[2, 1]
#print(chessboard)
#print(np.count_nonzero(chessboard[1]))
                       
