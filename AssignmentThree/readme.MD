Heavy_Queens_Tuple_Gen.py generartes (board, total path cost tuples)
Heavy_Queens_Data_Gen.py generates attributes given output of Heavy_Queens_Tuple_gen.py
Heavy_Queens_A_Star_With_Random_Forest_Heuristic.py is the heavy queens algorithm with the random forest model inserted and takes the board as input and returns a predicted future path cost.
board_scores.csv is the dataset of board, solution cost tuples.
attributes.csv is the dataset of features derived from board_scores.csv.
model.pkl is the trained random forest model.
Model_Train.ipynb is the notebook of training different models and finding the best one.
Heavy_Queens_Data_Analysis_Notebook is the notebook of exploring our attributes.csv dataset.
