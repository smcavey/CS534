import numpy as np
import time

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

chessboard = np.array([[4, 0, 0, 0, 9, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0, 4, 0],
                       [0, 0, 0, 8, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 4],
                       [0, 0, 0, 0, 0, 0, 0]])

'''chessboard = np.array([[4, 0, 0, 0, 9],
                       [0, 0, 0, 0, 0],
                       [0, 2, 8, 0, 0],
                       [0, 0, 0, 8, 0],
                       [0, 0, 0, 0, 0]])'''


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
    return current_cost 

def run_Astar(chessboard, fringe_chess):
    chosen_chess = np.argmin(fringe_chess[:, 1])
    for i in range(0, len(chessboard)):
        for j in range(0, len(chessboard)):
            if fringe_chess[chosen_chess, 0][j, i] > 0:
                for k in list(range(0,j)) + list(range(j+1,len(chessboard))):
                    temp_chess = np.copy(fringe_chess[chosen_chess, 0])
                    temp_chess[j, i], temp_chess[k, i] = temp_chess[k, i], temp_chess[j, i]
                    cost = abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + get_current_cost(temp_chess) + fringe_chess[chosen_chess, 2]
                    fringe_chess = np.append(fringe_chess, np.array([[temp_chess, cost, abs(j - k) * (fringe_chess[chosen_chess, 0][j, i] ** 2) + fringe_chess[chosen_chess, 2], get_current_cost(temp_chess), fringe_chess[chosen_chess, 4] + "Move {},{} to {},{}. ".format(j, i, k, i)]]), axis=0)
    
    fringe_chess = np.delete(fringe_chess, chosen_chess, 0)
    
    '''Comment out if you only wanna print the final solution'''
    print(fringe_chess)
    
    '''Checks to see if a solution is found and returns the min solution if there is a solution.'''
    if 0 in fringe_chess[:, 3]:
        print("ANSWER:")
        answer = np.where(fringe_chess[:, 3] == 0)
        temp_cost = 10000000000
        temp_index = -1
        for i in answer[0]:
            b = fringe_chess[i, 1]
            if fringe_chess[i, 1] < temp_cost:
                temp_index = i
                temp_cost = fringe_chess[i, 1]
        
        
        return fringe_chess[temp_index]
    
        '''If no solution, restarts the search with all opened nodes in the fringe_chess array.'''
    else:
        return run_Astar(chessboard, fringe_chess)

'''Starts the search~'''
if __name__ == '__main__':
    '''fringe_chess = chessboard, total cost, past cost, future cost'''
    fringe_chess = np.array([[chessboard, 0, 0, get_current_cost(chessboard), ""]])
    print(run_Astar(chessboard, fringe_chess))


'''Notes for testing functions in case I need them'''
#print(np.any(chessboard[3, :] > 0))
#print(np.sum(chessboard[2]))
#chessboard[2, 1], chessboard[2, 2] = chessboard[2, 2], chessboard[2, 1]
#print(chessboard)
#print(np.count_nonzero(chessboard[1]))
                       