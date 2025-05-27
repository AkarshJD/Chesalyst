import numpy as np
from board import Board
from evaluation import Evaluator
from search import Searcher

def print_evaluation_bar(score):
    total_slots = 20
    normalized_score = max(min(score, 5), -5)  # clamp to [-5, 5] for display
    slots = int((normalized_score + 5) / 10 * total_slots)

    bar = '[' + '#' * slots + '-' * (total_slots - slots) + ']'
    if score > 0:
        advantage = f"+{score:.2f}"
    else:
        advantage = f"{score:.2f}"
    print(f"{bar} {advantage}")


def choose_time_control():
    print("Choose time control:")
    print("1. 5+3")
    print("2. 5+5")
    print("3. 15+10")
    print("4. 25+15")
    print("5. 60+30")
    print("6. 120+30")
    choice = int(input("Enter choice: "))

    time_controls = {
        1: (5*60, 3),
        2: (5*60, 5),
        3: (15*60, 10),
        4: (25*60, 15),
        5: (60*60, 30),
        6: (120*60, 30)
    }
    base_time, increment = time_controls.get(choice, (5*60, 3))

    show_thinking = input("Show engine thinking (y/n)? ").lower() == 'y'

    # replace current side-choice section
    side_choice = input("Play as (w)hite, (b)lack, or (a)nalysis? ").lower()
    if side_choice == 'w':
        player_is_white = True
        analysis_mode = False
    elif side_choice == 'b':
        player_is_white = False
        analysis_mode = False
    else:           # analysis default
        player_is_white = True
        analysis_mode = True

    return base_time, increment, show_thinking, player_is_white, analysis_mode

def square_to_index(square):
    files = 'abcdefgh'
    ranks = '12345678'
    if len(square) != 2:
        return None
    file = files.find(square[0])
    rank = ranks.find(square[1])
    if file == -1 or rank == -1:
        return None
    return (7 - rank) * 8 + file

def parse_move(move_str: str, board: Board):
    """
    Convert user text into a *legal* Move object.

    Rules we enforce (Stockfish‑style):

    ─ 4‑ or 5‑character long algebraic only, e.g.  **d2d4  g7g8q**
      (the 5th char is an optional promotion piece).

    ─ Castling keywords:  O‑O  /  O‑O‑O  (also accepts 0‑0 / 0‑0‑0).

    Any other input is rejected with a short hint so the calling
    code can ask the user to try again.
    """
    move_str = move_str.strip()
    if move_str.lower() == "exit":
        print("Exiting the game.");  exit(0)

    # --- handle castling --------------------------------------------------
    if move_str.lower() in ("o-o", "0-0"):
        for mv in board.generate_legal_moves():
            if mv.is_kingside_castle():
                return mv
        return None

    if move_str.lower() in ("o-o-o", "0-0-0"):
        for mv in board.generate_legal_moves():
            if mv.is_queenside_castle():
                return mv
        return None
    # ---------------------------------------------------------------------

    # --- strip optional "=q" style promotion ------------------------------
    promotion_piece = None
    if '=' in move_str:
        move_str, promotion_piece = move_str.split('=', 1)
    elif len(move_str) == 5:                 # d7d8q
        promotion_piece, move_str = move_str[-1], move_str[:-1]
    # ---------------------------------------------------------------------

    # --- long algebraic only from here -----------------------------------
    if len(move_str) != 4:
        print("Please enter moves in long form, e.g.  d2d4  or  e7e8q.")
        return None

    files = 'abcdefgh'
    ranks = '12345678'

    def sq_index(txt: str) -> int | None:
        file_idx = files.find(txt[0])
        rank_idx = ranks.find(txt[1])
        if file_idx == -1 or rank_idx == -1:
            return None
        return (7 - rank_idx) * 8 + file_idx

    from_sq = sq_index(move_str[:2])
    to_sq   = sq_index(move_str[2:])

    if from_sq is None or to_sq is None:
        return None

    for mv in board.generate_legal_moves():
        if mv.from_square == from_sq and mv.to_square == to_sq:
            if promotion_piece:
                mv.promotion = (promotion_piece.upper()
                                 if board.white_to_move
                                 else promotion_piece.lower())
            return mv

    # If we get here the text didn't map to a legal move
    print("Illegal or ambiguous move; try again with long notation.")
    return None

def format_move(move):
    files = 'abcdefgh'
    ranks = '12345678'
    from_file = files[move.from_square % 8]
    from_rank = ranks[7 - move.from_square // 8]
    to_file = files[move.to_square % 8]
    to_rank = ranks[7 - move.to_square // 8]

    # Castling
    if move.is_kingside_castle():
        return 'O-O'
    if move.is_queenside_castle():
        return 'O-O-O'

    move_str = ''
    if move.is_pawn_move():
        if move.is_capture():
            move_str += from_file
            move_str += 'x'
        move_str += to_file + to_rank
        if move.promotion:
            move_str += '=' + move.promotion.upper()
    else:
        # For pieces like Knight, Bishop etc (only Knight for now due to simple Move class)
        move_str += from_file + from_rank
        if move.is_capture():
            move_str += 'x'
        move_str += to_file + to_rank
    return move_str

def pretty_print_board(board, player_is_white):
    if player_is_white:
        file_labels = 'a b c d e f g h'.split()
        rank_labels = list(range(8, 0, -1))        # show 8→1
        rows = range(0, 8)                         # internal rows 0(a8)…7(a1)
        file_iter = range(0, 8)
    else:
        file_labels = 'h g f e d c b a'.split()
        rank_labels = list(range(1, 9))            # show 1→8 from Black’s pov
        rows = range(7, -1, -1)                    # start from a1‑row (index 7)
        file_iter = range(7, -1, -1)

    print('    ' + '  '.join(file_labels))
    for r_idx, r in enumerate(rows):
        cells = []
        for f in file_iter:
            sq = r * 8 + f
            piece_char = '·' if (r + f) % 2 else '•'
            for piece, bb in board.bitboards.items():
                if (bb >> np.uint64(sq)) & np.uint64(1):
                    is_white_piece = piece.isupper()
                    own_piece = (is_white_piece and player_is_white) or (not is_white_piece and not player_is_white)
                    color = '\033[97m' if own_piece else '\033[30m'
                    piece_char = f"{color}{piece}\033[0m"
                    break
            cells.append(piece_char)
        print(f"{rank_labels[r_idx]} | {'  '.join(cells)} | {rank_labels[r_idx]}")
    print('    ' + '  '.join(file_labels))

def is_insufficient_material(board):
    pieces = []
    for piece, bb in board.bitboards.items():
        count = bin(bb).count('1')
        pieces.extend([piece] * count)
    pieces = [p.lower() for p in pieces]

    if pieces.count('k') == 2 and len(pieces) == 2:
        return True
    if pieces.count('k') == 2 and pieces.count('b') == 1 and len(pieces) == 3:
        return True
    if pieces.count('k') == 2 and pieces.count('n') == 1 and len(pieces) == 3:
        return True
    return False

def main():
    board = Board()
    evaluator = Evaluator()
    base_time, increment, show_thinking, player_is_white, analysis_mode = choose_time_control()
    searcher = Searcher(evaluator, base_time=base_time, increment=increment, show_thinking=show_thinking)
    move_number = 1
    pgn_moves = []
    position_counts = {}
    halfmove_clock = 0
    # engine will move automatically when it's the opposite side and we’re not in analysis
    auto_engine = not analysis_mode

    while True:
        pretty_print_board(board, player_is_white)
        # If it's engine's turn and auto play enabled, let engine think & move
        if auto_engine and (board.white_to_move != player_is_white):
            print("Engine is thinking...")
            engine_move = searcher.search(board, move_number)
            # Print engine's evaluation of each candidate at root
            root_list = sorted(searcher.root_scores, key=lambda x: x[0], reverse=board.white_to_move)
            mv_strings = []
            for s, mv in root_list:
                tag = "*" if mv == engine_move else " "
                mv_strings.append(f"{tag}{format_move(mv)}({s/100:+.2f})")
            print("Engine candidates:", ' '.join(mv_strings))
            print(f"Engine plays: {format_move(engine_move)}")
            board.make_move(engine_move)
            move_number += 1
            continue

        score = evaluator.evaluate(board) / 100  # Convert centipawns to pawns
        print_evaluation_bar(score)

        if board.white_to_move:
            print("White to move")
        else:
            print("Black to move")

        # Show each legal move with its static evaluation
        valid_moves = board.generate_legal_moves()
        move_eval_strs = []
        for mv in valid_moves:
            board.make_move(mv)
            mv_score = evaluator.evaluate(board) / 100  # centipawns → pawns
            board.undo_move(mv)
            move_eval_strs.append(f"{format_move(mv)}({mv_score:+.2f})")
        print("Valid moves:", ' '.join(move_eval_strs))

        user_input = input("Enter your move: ").lower().strip()
        if any(word in user_input for word in ['draw', 'daw', 'drwa']):
            user_input = 'draw'
        elif any(word in user_input for word in ['resign', 'resing', 'resgn', 'resgin']):
            user_input = 'resign'
        elif any(word in user_input for word in ['engine', 'engin', 'egnine', 'engien']):
            user_input = 'engine'
        if user_input == "draw":
            eval_score = evaluator.evaluate(board) / 100
            if -1 <= eval_score <= 1:
                print("Engine accepts your draw offer.")
                print_pgn(pgn_moves, result="1/2-1/2")
                return
            else:
                print("Engine declines the draw. Play continues.")
                continue
        if user_input == "resign":
            print("You resigned. Game over.")
            print_pgn(pgn_moves)
            return

        if user_input == "engine":
            eval_score = evaluator.evaluate(board) / 100
            if -1 <= eval_score <= 1:
                print("Engine offers a draw based on balanced evaluation.")
                accept_draw = input("Accept draw (y/n)? ").lower()
                if accept_draw == 'y':
                    print("Draw agreed.")
                    print_pgn(pgn_moves)
                    return
            move = searcher.search(board, move_number)
            if move:
                print(f"Engine plays: {format_move(move)}")
                board.make_move(move)
                pgn_moves.append(format_move(move))

                move_hash = hash(str(board.bitboards) + str(board.white_to_move))
                position_counts[move_hash] = position_counts.get(move_hash, 0) + 1

                if move.is_capture() or move.is_pawn_move():
                    halfmove_clock = 0
                else:
                    halfmove_clock += 1

                # Check for threefold repetition
                if position_counts[move_hash] >= 3:
                    print("Threefold repetition detected. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for 50-move rule
                if halfmove_clock >= 100:
                    print("50-move rule draw detected.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for insufficient material
                if is_insufficient_material(board):
                    print("Insufficient material. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for stalemate
                if not board.generate_moves() and not board.is_in_check():
                    print("Stalemate detected. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                move_number += 1
            else:
                print("No move found.")
        else:
            move = parse_move(user_input, board)
            if move:
                board.make_move(move)
                pgn_moves.append(format_move(move))

                move_hash = hash(str(board.bitboards) + str(board.white_to_move))
                position_counts[move_hash] = position_counts.get(move_hash, 0) + 1

                if move.is_capture() or move.is_pawn_move():
                    halfmove_clock = 0
                else:
                    halfmove_clock += 1

                # Check for threefold repetition
                if position_counts[move_hash] >= 3:
                    print("Threefold repetition detected. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for 50-move rule
                if halfmove_clock >= 100:
                    print("50-move rule draw detected.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for insufficient material
                if is_insufficient_material(board):
                    print("Insufficient material. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                # Check for stalemate
                if not board.generate_moves() and not board.is_in_check():
                    print("Stalemate detected. Game drawn.")
                    print_pgn(pgn_moves, result="1/2-1/2")
                    save_pgn(pgn_moves, result="1/2-1/2")
                    return

                move_number += 1
            else:
                print("Invalid move, try again.")

def print_pgn(pgn_moves, result=None):
    save_pgn(pgn_moves, result)

def save_pgn(pgn_moves, result=None):
    with open("last_game.pgn", "w") as f:
        f.write('[Event "Friendly Match"]\n')
        f.write('[Site "Local Engine"]\n')
        f.write(f'[Result "{result if result else "*"}"]\n\n')
        for i in range(0, len(pgn_moves), 2):
            move_number = i // 2 + 1
            white_move = pgn_moves[i]
            black_move = pgn_moves[i+1] if i+1 < len(pgn_moves) else ''
            f.write(f"{move_number}. {white_move} {black_move} ")
        if result:
            f.write(f"\n{result}\n")

if __name__ == "__main__":
    main()
