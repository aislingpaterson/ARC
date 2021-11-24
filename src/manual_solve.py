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


def solve_681b3aeb(x):
    ''' Assumptions made for this task:
      - background for inputs is always black (colour code 0)
      - orientation of shapes is always correct for combination (rotation of shapes not required)
      - output is always 3x3 
      - always involves combining 2 shapes, not more - and shapes completely fill the 3x3 output
      - the two shapes always have distinct colours
    
    Transformation carried out: 
    1. The set of distinct colours in the grid is determined, and the background colour removed from this set.
    2. The grid is attempted to be split in two, to achieve each of the shapes in an individual array.
    If the split is carried out and one of the resulting arrays is empty (all background), the split is repeated for the non-empty array.
    Otherwise, the two arrays are tested to ensure that each shape has been partitioned fully into each of the resultant arrays.
    This is done by ensuring the same colour isn't present across both arrays.
    3. If the arrays fail this task, the split is re-performed on a different axis on the grid. 
    All the different possible splitting axes (in the x and y directions) are iterated over until
    a working split has been found.
    4. Then, rows and columns consisting solely background colour (black) are removed from the two arrays.
    5. Next, if the resulting arrays are less than the output shape (3x3), they are padded to reach this shape.
    We do not know how these shapes should be padded, so padding is repeatedly randomly performed until 
    the padded 3x3 shape and the other shape (possibly also needing randomised padding), add together to give an array with no 
    background elements present. Since the set of possible shapes which need padding (such as 2x2, 3x2, 1x2 arrays, etc) 
    is quite small - the randomised padding works efficiently enough. I felt this was more efficient than hardcoding all possible paddings for all possible shapes.
    6. The array which the two individual arrays combine with no background elements, 
    is returned as the output.
    
    All of the training/test grids appear to solve succesfully with this function. 
    '''
    # set these variables as per the assumptions made
    background_colour = 0
    x_out_shape, y_out_shape = 3,3
    
    rows,cols = x.shape
    
    # determine distinct colours 
    distinct_colours = set(x.flatten().tolist())
    distinct_colours.remove(background_colour)

    def check_colour_split(x1,x2,distinct_colours=distinct_colours):
        '''Check that each shape has been partitioned fully in the split arrays by 
        checking that the same colour does not exist across the two arrays'''
        for colour in distinct_colours:
            if colour in x1 and colour in x2:
                return False
        return True
    
    def check_empty(array,background_colour=background_colour):
        '''Check if all row elements are the background colour'''
        return np.all(np.all(array == background_colour, axis=1) == True)
    
    def perform_splitting(x, original_grid=np.copy(x), axis=0, split_position=False):
        '''split grid in half - if one half is empty -
        recursively call splitting again with other half'''

        # set default splitting position as half way along the axis we are splitting on
        if not split_position:
            split_position = [x.shape[axis]//2]
        
        split_grid = np.array_split(x,split_position,axis)
        
        #check if one of split pieces is empty - if so, recursively call function with non-empty portion
        
        for i in range(len(split_grid)):
            if check_empty(split_grid[i]):
                if check_empty(split_grid[i-1]):
                    raise ValueError('Whole grid is empty (all background colour)')
                else: return perform_splitting(split_grid[i-1],original_grid)
        
        # check if grid is correctly colour separated - if not try a y-axis split
        
        if check_colour_split(*split_grid):
            return split_grid
        else:   
            return False
        
    def run_splitting(x):
        '''run perform splitting function, if grid isn't usefully split, run again with 
        alternative split positions/split axes '''
        output = perform_splitting(x)
        if not output:
            for i in range(1,x.shape[1]):
                output = perform_splitting(x, axis=1, split_position=[i])
                if output:
                    return output
            if not output:
                for j in range(1,x.shape[0]):
                    output = perform_splitting(x, axis=0, split_position=[j])
                    if output:
                        return output 
            if not output:
                return 'Only one colour exists in grid'
        else: return output
    
    def strip_background(x):
        # remove rows that are all background, and then transpose to repeat for columns
        x = x[~np.all(x == 0, axis=1)]
        x = x.T[~np.all(x.T == 0, axis=1)]
        #return to original orientation
        return x.T        
                
    def transform_to_output_shape(x, x_out_shape=x_out_shape, y_out_shape=y_out_shape):
        while x.shape != (x_out_shape,y_out_shape):
            if x.shape[0] != x_out_shape:
                b = np.zeros(x.shape[1])
                xb = [x,b]
                random.shuffle(xb)
                x = np.vstack((xb))
            if x.shape[1] != y_out_shape:
                a = np.zeros(x.shape[0]).reshape(-1,1)
                xa = [x,a]
                random.shuffle(xa)
                x = np.hstack((xa))
        return x

                
    x1,x2 = run_splitting(x)
    x1 = strip_background(x1)
    x2 = strip_background(x2)
    
    # create empty output array
    y_hat = np.zeros((x_out_shape,y_out_shape))
    
    # add the two arrays until they correctly fit together (if not - change padding on shape smaller than 3x3)
    while np.any(np.any(y_hat == background_colour, axis=1) == True):
        x1_new = transform_to_output_shape(x1)
        x2_new = transform_to_output_shape(x2)
        y_hat = x1_new + x2_new

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
