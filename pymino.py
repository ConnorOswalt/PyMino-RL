import numpy as np
import random
import os
import datetime

# requirements: numpy>=1.18

# [GLOBALS]
rows_and_cols = 9

# Create board

def create_board():
    return np.zeros((rows_and_cols, rows_and_cols), dtype=np.int8)

# Save endgame counts

def save_board_snapshot(board, csv_path='endgame_board.csv'):
    if os.path.exists(csv_path):
        counts = np.loadtxt(csv_path, delimiter=',', dtype=int)
        if counts.shape != (rows_and_cols, rows_and_cols):
            counts = np.zeros((rows_and_cols, rows_and_cols), dtype=int)
    else:
        counts = np.zeros((rows_and_cols, rows_and_cols), dtype=int)
    counts += (board != 0).astype(int)
    np.savetxt(csv_path, counts, fmt='%d', delimiter=',')
    #print(f"Endgame counts updated in {csv_path}")

# Morphisms
def random_transform(mask):
    m = np.rot90(mask, k=random.choice((0,1,2,3)))
    return np.fliplr(m) if random.choice((True,False)) else m

def draw_shapes(base_shapes, k=3):
    names = random.choices(list(base_shapes), k=k)
    return [random_transform(base_shapes[n]) for n in names]

# Placement checks
def can_place(board, shape, r, c):
    h,w = shape.shape
    if r<0 or c<0 or r+h>rows_and_cols or c+w>rows_and_cols:
        return False
    return not np.any((board[r:r+h, c:c+w]!=0)&shape)

def all_valid_positions(board, shape):
    h,w = shape.shape
    positions=[]
    for r in range(rows_and_cols-h+1):
        for c in range(rows_and_cols-w+1):
            if can_place(board, shape, r, c):
                positions.append((r,c))
    return positions

# Clearing routines
def clear_full_rows(board):
    pts=0
    full = np.where(board.sum(axis=1)==rows_and_cols)[0]
    for r in full:
        board[r,:]=0; pts+=rows_and_cols*2
    return pts

def clear_full_cols(board):
    pts=0
    full = np.where(board.sum(axis=0)==rows_and_cols)[0]
    for c in full:
        board[:,c]=0; pts+=rows_and_cols*2
    return pts

def clear_full_blocks(board):
    pts=0
    for br in range(0,rows_and_cols,3):
        for bc in range(0,rows_and_cols,3):
            blk=board[br:br+3, bc:bc+3]
            if np.all(blk!=0):
                board[br:br+3, bc:bc+3]=0; pts+=3*3*2
    return pts

# Base shapes
BASE_SHAPES={
 'single':np.array([[1]],bool),
 'domino':np.array([[1,1]],bool),
 'trio':np.array([[1,1,1]],bool),
 'tetramino':np.array([[1,1,1,1]],bool),
 'L':np.array([[1,0],[1,1]],bool),
 'BIG L':np.array([[1,0,0],[1,0,0],[1,1,1]],bool),
 'Diag-2':np.array([[0,1],[1,0]],bool),
 'Diag-3':np.array([[1,0,0],[0,1,0],[0,0,1]],bool),
 'Lanky L':np.array([[1,0],[1,0],[1,1]],bool),
 'Cross':np.array([[0,1,0],[1,1,1],[0,1,0]],bool),
 'C':np.array([[1,1],[1,0],[1,1]],bool),
 'T':np.array([[0,1,0],[1,1,1]],bool),
 'S':np.array([[1,1,0],[0,1,1]],bool),
 'square':np.array([[1,1],[1,1]],bool)
}

# Automatic gameplay

def main():
        board=create_board()
        score=0
        while True:
            shapes=draw_shapes(BASE_SHAPES,3)
            # Try to place all three
            for shape in shapes:
                poss=all_valid_positions(board,shape)
                if not poss:
                    # no valid for this shape
                    save_board_snapshot(board)
                    print('Game Over! Score:',score)
                    return
                r,c=random.choice(poss)
                # place and score
                board[r:r+shape.shape[0], c:c+shape.shape[1]]+=shape.astype(int)
                score+=int(shape.sum())
                # clears
                pts=clear_full_rows(board)+clear_full_cols(board)+clear_full_blocks(board)
                score+=pts
            # loop new shapes

if __name__=='__main__': main()
