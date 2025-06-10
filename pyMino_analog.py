import numpy as np
import random
import os

#CLI driver for base pymino.py


# requirements: numpy>=1.18

# [GLOBALS]
rows_and_cols = 9

# Utility to clear console screen

def clear_screen():
    """Clear terminal screen on Windows or Unix-like systems."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Create and print board with perforations

def create_board():
    """Initialize a 9×9 Blockudoku game board as a NumPy array."""
    return np.zeros((rows_and_cols, rows_and_cols), dtype=np.int8)

def print_board(board):
    """Print the board with 3×3 perforations: '░' for empty, '█' for occupied."""
    n = rows_and_cols
    tokens_count = n + (n - 1) // 3
    inner_width = tokens_count * 2 - 1
    # Top border
    print("┌" + "─" * inner_width + "┐")
    for r_idx, row in enumerate(board):
        tokens = []
        for c_idx, cell in enumerate(row):
            tokens.append('█' if cell else '░')
            if c_idx % 3 == 2 and c_idx != n - 1:
                tokens.append('|')
        print("│" + ' '.join(tokens) + "│")
        if r_idx % 3 == 2 and r_idx != n - 1:
            print("│" + "." * inner_width + "│")
    # Bottom border
    print("└" + "─" * inner_width + "┘")

# Base shape definitions
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

def random_transform(shape_mask):
    """Randomly rotate and optionally mirror a shape mask."""
    rotated = np.rot90(shape_mask, k=random.choice((0, 1, 2, 3)))
    if random.choice((True, False)):
        rotated = np.fliplr(rotated)
    return rotated

def draw_shapes(base_shapes, k=3):
    """Draw k random shapes with random orientation and flip."""
    names = random.choices(list(base_shapes), k=k)
    return [random_transform(base_shapes[name]) for name in names]

# Placement validity checks

def can_place(board, shape, r, c):
    """Check if shape can be placed at (r,c) without mutating the board."""
    h, w = shape.shape
    if r < 0 or c < 0 or r + h > rows_and_cols or c + w > rows_and_cols:
        return False
    sub = board[r:r+h, c:c+w]
    return not np.any((sub != 0) & shape)

def can_place_shape(board, shape):
    """Return True if shape can be placed anywhere on the board."""
    h, w = shape.shape
    for r in range(rows_and_cols - h + 1):
        for c in range(rows_and_cols - w + 1):
            if can_place(board, shape, r, c):
                return True
    return False

# Clearing functions with scoring

def clear_full_rows(board):
    """Clear full rows and return points (2 per cell)."""
    full = np.where(board.sum(axis=1) == rows_and_cols)[0]
    pts = 0
    for r in full:
        board[r, :] = 0
        pts += rows_and_cols * 2
    return pts

def clear_full_cols(board):
    """Clear full columns and return points (2 per cell)."""
    full = np.where(board.sum(axis=0) == rows_and_cols)[0]
    pts = 0
    for c in full:
        board[:, c] = 0
        pts += rows_and_cols * 2
    return pts

def clear_full_blocks(board):
    """Clear filled 3×3 subgrids and return points (2 per cell)."""
    pts = 0
    for br in range(0, rows_and_cols, 3):
        for bc in range(0, rows_and_cols, 3):
            block = board[br:br+3, bc:bc+3]
            if np.all(block != 0):
                board[br:br+3, bc:bc+3] = 0
                pts += 3 * 3 * 2
    return pts

# Shape display function

def print_shapes(board, shapes):
    """Print shapes side-by-side with '█' and '░', label unplaceable with 'x'."""
    heights = [s.shape[0] for s in shapes]
    widths = [s.shape[1] for s in shapes]
    max_h = max(heights)
    labels, col_w = [], []
    for i, (s, w) in enumerate(zip(shapes, widths)):
        ok = can_place_shape(board, s)
        lbl = f"{i+1}{'' if ok else 'x'}"
        cw = max(2*w-1, len(lbl))
        labels.append(lbl.center(cw))
        col_w.append(cw)
    print('   '.join(labels))
    for r in range(max_h):
        row_strs = []
        for i, s in enumerate(shapes):
            h, w = heights[i], widths[i]
            if r < h:
                row = ' '.join('█' if cell else '░' for cell in s[r])
            else:
                row = '░ ' * w; row = row.strip()
            row_strs.append(row.center(col_w[i]))
        print('   '.join(row_strs))
    print()

# Place shape onto board

def place_shape(board, shape, r, c):
    """Place shape on board at (r,c). Assumes valid."""
    h, w = shape.shape
    board[r:r+h, c:c+w] += shape.astype(board.dtype)

# Main game loop

def main():
    board = create_board()
    shapes = draw_shapes(BASE_SHAPES, k=3)
    score = 0
    while True:
        # Clear and redraw screen
        clear_screen()
        # Check for end-game
        if not any(can_place_shape(board, s) for s in shapes):
            print("\nGame Over! No valid moves remain.")
            print("Final Board:")
            print_board(board)
            print(f"Score: {score}\n")
            print("Remaining shapes:")
            print_shapes(board, shapes)
            break
        # Display current state
        print("\nCurrent Board:")
        print_board(board)
        print(f"Score: {score}\n")
        print_shapes(board, shapes)
        # User selects a shape
        while True:
            ch = input("Select shape (1-3): ")
            if ch in ('1','2','3'):
                idx = int(ch) - 1
                if idx < len(shapes) and can_place_shape(board, shapes[idx]):
                    shape = shapes.pop(idx)
                    break
            print("Invalid, pick 1-3 on placeable shapes.")
        # User selects coordinates
        while True:
            coord = input("Enter row,col (0-8): ")
            try:
                r, c = map(int, coord.split(','))
            except:
                print("Use row,col format.")
                continue
            if can_place(board, shape, r, c):
                place_shape(board, shape, r, c)
                # Score placement
                score += int(shape.sum())
                # Apply clears
                pts = (clear_full_rows(board) +
                       clear_full_cols(board) +
                       clear_full_blocks(board))
                if pts > 0:
                    score += pts
                    print(f"Cleared! Gained {pts} points.")
                break
            else:
                print("Cannot place there.")
        # Refill shapes when empty
        if not shapes:
            shapes = draw_shapes(BASE_SHAPES, k=3)

if __name__ == '__main__':
    main()