import os, sys
import json
import numpy as np
import re
import itertools
import random

'''
Aisling Paterson, student no. 21249294
'''

def solve_5ad4f10b(x):
    '''  Assumptions made for this task:
     - Background colour for this task is always black (colour code 0).
     - Always outputs to a 3x3 grid.
     - The shape we wish to emulate in the output will always have a width greater 
       than one grid unit, otherwise it is 'noise'. 
       
       Transformation carried out:
       1. The 'noise' is detected and wiped from the grid, with the noise colour being identified 
       as the first row-wise element to be detected which has a different colour to neighbouring row elements, as per assumptions above.
       2. Rows and columns consisting solely background colour (black) are removed.
       3. The resulting array shape is mapped to the 3x3 grid. The resulting array is assumed to divide in 3 in both dimensions for this mapping, 
       and this is performed to carry out the mapping.
       
    All of the training/test grids appear to solve succesfully with this function. 
    '''
    
    background_colour = 0
    x_out_shape, y_out_shape = 3,3
    
    rows,cols = x.shape
    
    # search for noise in dataset
    for row in x:
        for i in range(cols-1):
            # if element is not background colour, and next element 
            # and the element before is not the same colour - it must be noise
            # from assumption that width of shape to be emulated is greater than one unit in width
            if row[i] != 0 and row[i+1] != row[i] and row[i-1] != row[i]:
                noise_colour = row[i]
                break
                
    # strip noise from array - replace with background colour
    x = np.where(x == noise_colour, background_colour, x)
    
    # remove rows that are all background, and then transpose to repeat for columns
    x = x[~np.all(x == 0, axis=1)]
    x = x.T[~np.all(x.T == 0, axis=1)]
    #return to original orientation
    x = x.T
    
    # determine pared-down shape of x
    rows_new,cols_new = x.shape
    # map onto 3x3 square - current grid must be divisible by 3
    rows_new //= x_out_shape
    cols_new //= y_out_shape

    # transform onto 3x3 grid
    output_list = []
    index_list = []
    # pick indexes which we want to transform to grid
    for i in range(0,x.shape[0],rows_new):
        for j in range(0,x.shape[1],cols_new):
                index_list.append((i,j))
                
    # create output with index list           
    for i,j in index_list:
        output_list.append(x[(i,j)])
    
    y_hat = np.array(output_list)
    y_hat = y_hat.reshape((3,3))

    # switch colour to correct colour - noise colour
    y_hat = np.where(y_hat != background_colour, noise_colour, y_hat)
    
    return y_hat


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()