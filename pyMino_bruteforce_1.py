#!/usr/bin/env python3
import json
import os
import argparse
from datetime import datetime, timezone
import numpy as np
from itertools import permutations

from pymino import (
    create_board,
    draw_shapes,
    all_valid_positions,
    clear_full_rows,
    clear_full_cols,
    clear_full_blocks,
    BASE_SHAPES,
)

# Default threshold beyond which we switch to greedy fallback
MAX_BRUTE_FORCE_SIMULATIONS = 200_000
VERSION="V1.0"

def print_board(board: np.ndarray):
    """Prints the 9×9 board: '·' for 0, '█' for 1."""
    for row in board:
        print(''.join('█' if cell else '·' for cell in row))
    print()

def place_and_score(board: np.ndarray, shape: np.ndarray, r: int, c: int):
    """
    Place `shape` at (r,c) on a copy of `board`, clear any full rows/cols/3×3 blocks,
    and return (new_board, points_gained).
    points_gained = shape.sum() + (2 points × each cleared cell).
    """
    new_board = board.copy()
    new_board[r : r + shape.shape[0], c : c + shape.shape[1]] += shape.astype(int)

    gain = int(shape.sum())
    gain += clear_full_rows(new_board)
    gain += clear_full_cols(new_board)
    gain += clear_full_blocks(new_board)
    return new_board, gain

def greedy_fallback_sequence(board: np.ndarray, shapes: list):
    """
    A fast “greedy‐sequential” fallback:
      • For each shape in index order (0→1→2), look at all valid (r,c),
        compute immediate gain = place_and_score(...)[1], pick the (r,c) with max gain,
        place it and move on.
    Returns:
      (total_gain, sequence_of_3 = [(shape_idx, r0,c0), (shape_idx, r1,c1), (shape_idx, r2,c2)])
      or (-999999, None) if any shape is unplaceable.
    """
    seq = []
    total_gain = 0
    temp_board = board.copy()

    for idx, mask in enumerate(shapes):
        valid = all_valid_positions(temp_board, mask)
        if not valid:
            return -999999, None

        best_gain = -1
        best_pos = None
        for (r, c) in valid:
            _, g = place_and_score(temp_board, mask, r, c)
            if g > best_gain:
                best_gain = g
                best_pos = (r, c)

        r0, c0 = best_pos
        temp_board, gain0 = place_and_score(temp_board, mask, r0, c0)
        total_gain += gain0
        seq.append((idx, r0, c0))

    return total_gain, seq

def find_best_sequence_for_three(board: np.ndarray, shapes: list):
    """
    Returns (best_total_gain, best_sequence = [(shape_idx, r0,c0), (shape_idx, r1,c1), (shape_idx, r2,c2)]).
    Behavior:
      1) If board.sum()==0 (completely empty), place shapes[0],shapes[1],shapes[2] each in their first valid pos.
      2) Otherwise, compute |valid1|, |valid2|, |valid3| on the original board. If
           6 × |v1| × |v2| × |v3| > MAX_BRUTE_FORCE_SIMULATIONS (and threshold > 0), run greedy_fallback.
      3) Else, do the exact brute force over all 6 permutations and all valid placements.
    If any shape has no valid placement up front, returns (-999999, None).
    """
    # --- Case 1: Board is completely empty
    if board.sum() == 0:
        seq = []
        total_gain = 0
        temp_board = board.copy()

        for idx, mask in enumerate(shapes):
            poss = all_valid_positions(temp_board, mask)
            if not poss:
                return -999999, None
            r0, c0 = poss[0]
            temp_board, g = place_and_score(temp_board, mask, r0, c0)
            total_gain += g
            seq.append((idx, r0, c0))

        return total_gain, seq

    # --- Case 2: Precompute valid positions on the original board
    valid_positions = []
    for mask in shapes:
        vp = all_valid_positions(board, mask)
        if not vp:
            # If any shape is immediately unplaceable, game over
            return -999999, None
        valid_positions.append(vp)

    v1, v2, v3 = map(len, valid_positions)
    total_simulations = 6 * v1 * v2 * v3

    # If MAX_BRUTE_FORCE_SIMULATIONS > 0 and brute-forcing would be too big, use greedy fallback
    if MAX_BRUTE_FORCE_SIMULATIONS > 0 and total_simulations > MAX_BRUTE_FORCE_SIMULATIONS:
        return greedy_fallback_sequence(board, shapes)

    # --- Case 3: Exact brute force
    best_total_gain = -1
    best_sequence = None

    for perm in permutations(range(3), 3):
        idx1, idx2, idx3 = perm

        for (r1, c1) in valid_positions[idx1]:
            b1, gain1 = place_and_score(board, shapes[idx1], r1, c1)

            valid2 = all_valid_positions(b1, shapes[idx2])
            if not valid2:
                continue

            for (r2, c2) in valid2:
                b2, gain2 = place_and_score(b1, shapes[idx2], r2, c2)

                valid3 = all_valid_positions(b2, shapes[idx3])
                if not valid3:
                    continue

                for (r3, c3) in valid3:
                    b3, gain3 = place_and_score(b2, shapes[idx3], r3, c3)
                    total_gain = gain1 + gain2 + gain3

                    if total_gain > best_total_gain:
                        best_total_gain = total_gain
                        best_sequence = [
                            (idx1, r1, c1),
                            (idx2, r2, c2),
                            (idx3, r3, c3),
                        ]

    return best_total_gain, best_sequence

def _dump_log(game_log, log_dir):
    """
    Write `game_log` as a JSON file named by game_id in log_dir,
    following naming scheme: score_<final_score>_<YYYY-MM-DD HH-MM-SS>.json
    """
    now = datetime.now()
    filename = os.path.join(
        log_dir,
        f"{VERSION}_score_{game_log['final_score']}_{now.strftime('%Y-%m-%d %H-%M-%S')}.json"
    )
    print(f"Trying to Save [Log] to {filename}")
    with open(filename, "w") as f:
        json.dump(game_log, f, indent=2)
    print(f"[Log] Saved game to {filename}")

def main(log_dir="logs"):
    # Ensure the logs folder exists (but maybe the user gave us something weird)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create log directory '{log_dir}': {e}")
        return

    board = create_board()
    total_score = 0
    turn_counter = 1

    # Prepare a game log dict with timezone‐aware UTC timestamp
    game_log = {
        "game_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "-autoplay",
        "final_score": None,
        "moves": []
    }

    while True:
        # 1) Snapshot board state before the turn
        board_before = board.copy().tolist()

        # 2) Draw 3 shapes
        shapes = draw_shapes(BASE_SHAPES, 3)

        # 3) Record “shapes in hand” masks
        shapes_info = []
        for idx, mask in enumerate(shapes):
            shapes_info.append({
                "shape_idx": idx,
                "mask": mask.astype(int).tolist()
            })

        # 4) Check immediate game-over: if any shape is unplaceable
        for idx, mask in enumerate(shapes):
            if not all_valid_positions(board, mask):
                game_log["final_score"] = total_score
                _dump_log(game_log, log_dir)
                print(f"Game Over! Final Score: {total_score}")
                return

        # 5) Find best sequence for this turn
        best_gain, best_seq = find_best_sequence_for_three(board, shapes)
        if best_seq is None:
            # If best_seq is None, treat as Game Over
            game_log["final_score"] = total_score
            _dump_log(game_log, log_dir)
            print(f"Game Over (no valid sequence)! Final Score: {total_score}")
            return

        # Announce which strategy was used
        if board.sum() == 0:
            print("Board was empty → first‐valid‐spot heuristic used (instant).")
        else:
            sizes = [len(all_valid_positions(board, s)) for s in shapes]
            if MAX_BRUTE_FORCE_SIMULATIONS > 0 and 6 * sizes[0] * sizes[1] * sizes[2] > MAX_BRUTE_FORCE_SIMULATIONS:
                print("Search space too large → using greedy‐sequential fallback.")
                print(
                    f"(|valid1|,|valid2|,|valid3|) = {tuple(sizes)}, "
                    f"6×prod = {6 * sizes[0] * sizes[1] * sizes[2]:,}"
                )
            else:
                print(f"Full brute-force found best immediate score = {best_gain} points")

        print(f"--- Turn {turn_counter} Sequence:")
        for idx, rr, cc in best_seq:
            print(f"  → Shape {idx} at (row={rr}, col={cc})")
        print()

        # 6) Execute the 3 placements, capturing per-move data
        turn_moves = []
        turn_score = 0
        for (shape_idx, r, c) in best_seq:
            shape = shapes[shape_idx]
            board, gain = place_and_score(board, shape, r, c)
            turn_score += gain
            total_score += gain

            turn_moves.append({
                "shape_idx": shape_idx,
                "r": r,
                "c": c,
                "gain": gain
            })

            print(f"Move (shape {shape_idx}) → placed at (r={r}, c={c}), "
                  f"gain = {gain}, total = {total_score}")
            print("Board now:")
            print_board(board)

        # 7) Append this turn’s info to game_log
        game_log["moves"].append({
            "turn": turn_counter,
            "board_before": board_before,
            "shapes": shapes_info,
            "chosen_sequence": turn_moves,
            "turn_score": turn_score
        })

        turn_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoplay the pymino game, adjust brute‐force threshold, and save logs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where JSON logs will be saved (default: ./logs)"
    )
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=200000,
        help=(
            "Threshold for full brute‐force (6×|v1|×|v2|×|v3|). "
            "If total_simulations > this value AND > 0, use greedy fallback. "
            "If set to 0, greedy fallback is disabled (always brute‐force)."
        )
    )
    args = parser.parse_args()

    # Override the global threshold according to the user's argument
    MAX_BRUTE_FORCE_SIMULATIONS = args.max_simulations

    main(log_dir=args.log_dir)
