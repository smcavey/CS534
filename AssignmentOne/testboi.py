import numpy as np

chessboard = np.array([ [0, 0, 0, 0 ,0, 0, 3],
                        [0, 0, 0, 0, 9, 0, 0,],
                        [0, 0, 4, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 7, 0],
                        [0, 3, 0, 0, 0, 0, 0],
                        [0, 0, 0, 5, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0]])

chessboard2 = np.array([[0, 0, 0, 4, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 3],
                        [0, 0, 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 2, 0],
                        [0, 2, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8, 0, 0]]) #one and 2

# chessboard3 = np.array([0 0 0 0 8 0 0]
# [0 0 0 0 0 0 3]
# [0 0 0 4 0 0 0]
# [0 0 5 0 0 0 0]
# [1 0 0 0 0 0 0]
# [0 0 0 0 0 2 0])

def find_upper_l(row,col,chessboard):
    #find uppermost lef diagonal
    while col >0 and row <0:
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
                #print(rightMost)
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
    moveCost = 0
    heuristic = 0
    data =[]

    #for every space
    for row in range(len(chessboard)):
        for col in range(len(chessboard)):
            if chessboard[row,col] !=0: #IF YOU FIND A QUEEN
                qInfo = [row,col,chessboard[row][col]] #row,col,weight
                queens.append(qInfo)
    queen = random.choice(queens) #pick one
    
    moveCost = queen[2]**2 
    #print(moveCost)
    move = make_random_move(chessboard,queen)
    #print(move)
    newChessboard = move[0]
    moveText = move[1]
    heuristic = spot_conflict(newChessboard)
    #print(heuristic)
    data.append(moveCost)
    data.append(heuristic)
    data.append(newChessboard)
    data.append(moveText)
    return data

if __name__ == '__main__':
    x = make_random_move(chessboard,[0,6,69])
    print(x)

    data = sample(x)
    # count = 0
    # x = spot_conflict(chessboard)
    # print(x)
    # y= spot_conflict(chessboard2)
    # print(y)

#check L diag
    # row= 2
    # col = 2
    # leftMost = find_upper_l(row,col,chessboard)
    # print(leftMost)
    # rowMax = leftMost[0]
    # colMax = leftMost[1]
    # for x in range(0,len(chessboard)-max(rowMax,colMax)):
    #     r= rowMax+x
    #     c= colMax+x
    #     if chessboard[r,c] != 0 and (r!=row and c!=col):
    #         count +=1



    #check R diag
    # count = 0
    # row = 0
    # col = 6
    # rightMost = find_upper_r(row,col,chessboard)
    # print(rightMost)
    # rowMax = rightMost[0]
    # colMax = rightMost[1]
    # for x in range(0,colMax-rowMax+1):
    #     r= rowMax+x
    #     c= colMax-x
    #     print([r,c])
    #     if chessboard[r,c] != 0 and (r!=row and c!=col):
    #         count +=1

    #print(count)
        