"""
Chesalyst — CLI entry point.

Run:
    pip install -e .
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from chesalyst.board import Board
from chesalyst.evaluation import Evaluator
from chesalyst.search import Searcher


# ------------------------------------------------------------------ #
#  Display helpers                                                     #
# ------------------------------------------------------------------ #

def print_eval_bar(score_cp):
    total = 20
    score_pawns = max(min(score_cp / 100, 5), -5)
    slots = int((score_pawns + 5) / 10 * total)
    bar = '[' + '#' * slots + '-' * (total - slots) + ']'
    tag = f"+{score_cp / 100:.2f}" if score_cp >= 0 else f"{score_cp / 100:.2f}"
    print(f"{bar} {tag}")


def pretty_board(board, player_is_white):
    if player_is_white:
        file_labels = list('abcdefgh')
        rank_labels = list(range(8, 0, -1))
        rows = range(0, 8)
        file_iter = range(0, 8)
    else:
        file_labels = list('hgfedcba')
        rank_labels = list(range(1, 9))
        rows = range(7, -1, -1)
        file_iter = range(7, -1, -1)

    print('    ' + '  '.join(file_labels))
    for r_idx, r in enumerate(rows):
        cells = []
        for f in file_iter:
            sq = r * 8 + f
            char = '·' if (r + f) % 2 else '•'
            for piece, bb in board.bitboards.items():
                if (bb >> np.uint64(sq)) & np.uint64(1):
                    own = (piece.isupper() and player_is_white) or (piece.islower() and not player_is_white)
                    color = '\033[97m' if own else '\033[30m'
                    char = f"{color}{piece}\033[0m"
                    break
            cells.append(char)
        print(f"{rank_labels[r_idx]} | {'  '.join(cells)} | {rank_labels[r_idx]}")
    print('    ' + '  '.join(file_labels))


def format_move(move):
    files = 'abcdefgh'
    ranks = '12345678'
    ff = files[move.from_square % 8]
    fr = ranks[7 - move.from_square // 8]
    tf = files[move.to_square % 8]
    tr = ranks[7 - move.to_square // 8]

    if move.is_kingside_castle():
        return 'O-O'
    if move.is_queenside_castle():
        return 'O-O-O'

    if move.is_pawn_move():
        s = (ff + 'x' if move.is_capture() else '') + tf + tr
        if move.promotion:
            s += '=' + move.promotion.upper()
        return s

    return ff + fr + ('x' if move.is_capture() else '') + tf + tr


# ------------------------------------------------------------------ #
#  Input parsing                                                       #
# ------------------------------------------------------------------ #

def parse_move(text, board):
    text = text.strip()
    if text.lower() == 'exit':
        print("Exiting.")
        sys.exit(0)

    if text.lower() in ('o-o', '0-0'):
        for m in board.generate_legal_moves():
            if m.is_kingside_castle():
                return m
        return None

    if text.lower() in ('o-o-o', '0-0-0'):
        for m in board.generate_legal_moves():
            if m.is_queenside_castle():
                return m
        return None

    promotion = None
    if '=' in text:
        text, promotion = text.split('=', 1)
    elif len(text) == 5:
        promotion, text = text[-1], text[:-1]

    if len(text) != 4:
        print("Use long algebraic notation, e.g. d2d4 or e7e8q.")
        return None

    files = 'abcdefgh'
    ranks = '12345678'

    def sq(t):
        fi = files.find(t[0])
        ri = ranks.find(t[1])
        return None if fi == -1 or ri == -1 else (7 - ri) * 8 + fi

    from_sq = sq(text[:2])
    to_sq = sq(text[2:])
    if from_sq is None or to_sq is None:
        return None

    for m in board.generate_legal_moves():
        if m.from_square == from_sq and m.to_square == to_sq:
            if promotion:
                m.promotion = promotion.upper() if board.white_to_move else promotion.lower()
            return m

    print("Illegal move. Try again.")
    return None


# ------------------------------------------------------------------ #
#  Draw / material helpers                                             #
# ------------------------------------------------------------------ #

def insufficient_material(board):
    pieces = []
    for piece, bb in board.bitboards.items():
        count = bin(int(bb)).count('1')
        pieces.extend([piece.lower()] * count)
    if len(pieces) == 2:
        return True
    if len(pieces) == 3 and (pieces.count('b') == 1 or pieces.count('n') == 1):
        return True
    return False


# ------------------------------------------------------------------ #
#  PGN                                                                 #
# ------------------------------------------------------------------ #

def save_pgn(pgn_moves, result=None):
    with open("last_game.pgn", "w") as f:
        f.write('[Event "Friendly Match"]\n')
        f.write('[Site "Local Engine"]\n')
        f.write(f'[Result "{result or "*"}"]\n\n')
        for i in range(0, len(pgn_moves), 2):
            wm = pgn_moves[i]
            bm = pgn_moves[i + 1] if i + 1 < len(pgn_moves) else ''
            f.write(f"{i // 2 + 1}. {wm} {bm} ")
        if result:
            f.write(f"\n{result}\n")
    print(f"Game saved to last_game.pgn")


# ------------------------------------------------------------------ #
#  Setup                                                               #
# ------------------------------------------------------------------ #

def choose_settings():
    print("Choose time control:")
    tc = {1: (300, 3), 2: (300, 5), 3: (900, 10), 4: (1500, 15), 5: (3600, 30), 6: (7200, 30)}
    for k, (b, i) in tc.items():
        print(f"  {k}. {b // 60}+{i}")
    try:
        choice = int(input("Choice: "))
        base, inc = tc.get(choice, (300, 3))
    except ValueError:
        base, inc = 300, 3

    show = input("Show engine thinking? (y/n): ").lower() == 'y'
    side = input("Play as (w)hite, (b)lack, or (a)nalysis mode? ").lower()
    player_white = side != 'b'
    analysis = side == 'a'
    return base, inc, show, player_white, analysis


# ------------------------------------------------------------------ #
#  Main game loop                                                      #
# ------------------------------------------------------------------ #

def main():
    board = Board()
    evaluator = Evaluator()
    base, inc, show, player_white, analysis = choose_settings()
    searcher = Searcher(evaluator, base_time=base, increment=inc, show_thinking=show)

    move_number = 1
    pgn_moves = []
    position_counts = {}
    halfmove = 0

    def record_position():
        h = hash(str({k: int(v) for k, v in board.bitboards.items()}) + str(board.white_to_move))
        position_counts[h] = position_counts.get(h, 0) + 1
        return position_counts[h]

    def check_draw_conditions(move):
        nonlocal halfmove
        if move.is_capture() or move.is_pawn_move():
            halfmove = 0
        else:
            halfmove += 1
        reps = record_position()
        if reps >= 3:
            return "Threefold repetition — draw."
        if halfmove >= 100:
            return "50-move rule — draw."
        if insufficient_material(board):
            return "Insufficient material — draw."
        return None

    while True:
        pretty_board(board, player_white)

        if board.is_checkmate():
            winner = "Black" if board.white_to_move else "White"
            print(f"Checkmate! {winner} wins.")
            result = "0-1" if board.white_to_move else "1-0"
            save_pgn(pgn_moves, result)
            return

        if board.is_stalemate():
            print("Stalemate — draw.")
            save_pgn(pgn_moves, "1/2-1/2")
            return

        # Engine turn (auto play when not analysis mode)
        engine_turn = not analysis and (board.white_to_move != player_white)
        if engine_turn:
            print("Engine thinking...")
            eng_move = searcher.search(board, move_number)
            cands = sorted(searcher.root_scores, key=lambda x: x[0], reverse=board.white_to_move)
            tags = [f"{'*' if m == eng_move else ' '}{format_move(m)}({s / 100:+.2f})"
                    for s, m in cands]
            print("Candidates:", ' '.join(tags))
            print(f"Engine plays: {format_move(eng_move)}")
            board.make_move(eng_move)
            pgn_moves.append(format_move(eng_move))
            draw_msg = check_draw_conditions(eng_move)
            if draw_msg:
                print(draw_msg)
                save_pgn(pgn_moves, "1/2-1/2")
                return
            move_number += 1
            continue

        # Human / analysis turn
        score = evaluator.evaluate(board)
        print_eval_bar(score)
        print("White to move" if board.white_to_move else "Black to move")

        valid = board.generate_legal_moves()
        evals = []
        for m in valid:
            board.make_move(m)
            evals.append(f"{format_move(m)}({evaluator.evaluate(board) / 100:+.2f})")
            board.undo_move(m)
        print("Moves:", ' '.join(evals))

        raw = input("Your move: ").lower().strip()

        # Typo-tolerant command matching
        if any(w in raw for w in ['draw', 'daw', 'drwa']):
            raw = 'draw'
        elif any(w in raw for w in ['resign', 'resing', 'resgn']):
            raw = 'resign'
        elif any(w in raw for w in ['engine', 'engin', 'egnine']):
            raw = 'engine'

        if raw == 'resign':
            print("You resigned.")
            save_pgn(pgn_moves, "0-1" if board.white_to_move else "1-0")
            return

        if raw == 'draw':
            if -100 <= score <= 100:
                print("Engine accepts the draw.")
                save_pgn(pgn_moves, "1/2-1/2")
                return
            print("Engine declines — position is unbalanced.")
            continue

        if raw == 'engine':
            eng_move = searcher.search(board, move_number)
            if eng_move:
                print(f"Engine plays: {format_move(eng_move)}")
                board.make_move(eng_move)
                pgn_moves.append(format_move(eng_move))
                draw_msg = check_draw_conditions(eng_move)
                if draw_msg:
                    print(draw_msg)
                    save_pgn(pgn_moves, "1/2-1/2")
                    return
                move_number += 1
            continue

        move = parse_move(raw, board)
        if move is None:
            continue

        board.make_move(move)
        pgn_moves.append(format_move(move))
        draw_msg = check_draw_conditions(move)
        if draw_msg:
            print(draw_msg)
            save_pgn(pgn_moves, "1/2-1/2")
            return
        move_number += 1


if __name__ == "__main__":
    main()
