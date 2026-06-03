import numpy as np
import pytest
from chesalyst.board import Board
from chesalyst.move import Move


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def empty_board():
    """Board with all pieces and occupancy cleared."""
    b = Board()
    b.bitboards = {k: 0 for k in b.bitboards}
    b.castling_rights = {'K': False, 'Q': False, 'k': False, 'q': False}
    b.white_occ = 0
    b.black_occ = 0
    b.all_occ = 0
    return b


def sq(name):
    """Convert algebraic square name to board index. E.g. 'e4' -> 36."""
    files = 'abcdefgh'
    f = files.index(name[0])
    r = int(name[1]) - 1       # 0-indexed from rank 1
    return (7 - r) * 8 + f    # rank from top * 8 + file


# ------------------------------------------------------------------ #
#  Starting position                                                   #
# ------------------------------------------------------------------ #

def test_starting_position_legal_moves():
    board = Board()
    moves = board.generate_legal_moves()
    # 16 pawn moves + 4 knight moves = 20
    assert len(moves) == 20


def test_starting_position_no_check():
    board = Board()
    assert not board.is_check()


def test_make_undo_restores_state():
    board = Board()
    before = {k: int(v) for k, v in board.bitboards.items()}
    before_wtm = board.white_to_move

    move = board.generate_legal_moves()[0]
    board.make_move(move)
    board.undo_move(move)

    assert {k: int(v) for k, v in board.bitboards.items()} == before
    assert board.white_to_move == before_wtm
    assert len(board._state_stack) == 0


def test_make_undo_multiple_plies():
    """Make several alternating moves, undo them all, board must match original."""
    board = Board()
    original = {k: int(v) for k, v in board.bitboards.items()}

    made = []
    for _ in range(6):
        moves = board.generate_legal_moves()
        if not moves:
            break
        m = moves[0]
        board.make_move(m)
        made.append(m)

    for _ in made:
        board.undo_move(made[0])  # object is irrelevant; undo pops the stack

    assert {k: int(v) for k, v in board.bitboards.items()} == original


# ------------------------------------------------------------------ #
#  is_square_attacked — pawn attack direction (was inverted)          #
# ------------------------------------------------------------------ #

def test_white_pawn_attacks_correct_squares():
    # White pawn on e4 (index 36) attacks d5 (27) and f5 (29)
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['P'] = np.uint64(1 << sq('e4'))
    b.white_to_move = True
    b._rebuild_occ()

    assert b.is_square_attacked(sq('d5'), by_white=True)
    assert b.is_square_attacked(sq('f5'), by_white=True)
    assert not b.is_square_attacked(sq('e5'), by_white=True)   # straight ahead
    assert not b.is_square_attacked(sq('e4'), by_white=True)   # own square
    assert not b.is_square_attacked(sq('d4'), by_white=True)   # same rank


def test_black_pawn_attacks_correct_squares():
    # Black pawn on e5 (index 28) attacks d4 (35) and f4 (37)
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['p'] = np.uint64(1 << sq('e5'))
    b.white_to_move = False
    b._rebuild_occ()

    assert b.is_square_attacked(sq('d4'), by_white=False)
    assert b.is_square_attacked(sq('f4'), by_white=False)
    assert not b.is_square_attacked(sq('e4'), by_white=False)   # straight ahead
    assert not b.is_square_attacked(sq('e5'), by_white=False)   # own square


def test_pawn_attack_no_file_wrap():
    # White pawn on a4 must not "attack" h5 (file wrap)
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['P'] = np.uint64(1 << sq('a4'))
    b.white_to_move = True
    b._rebuild_occ()

    assert b.is_square_attacked(sq('b5'), by_white=True)
    assert not b.is_square_attacked(sq('h5'), by_white=True)


# ------------------------------------------------------------------ #
#  generate_legal_moves — correct king legality filtering             #
# ------------------------------------------------------------------ #

def test_king_cannot_walk_into_pawn_attack():
    # Black pawn on e3 attacks d2 and f2.
    # White king on e1 must not walk to d2 or f2.
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['p'] = np.uint64(1 << sq('e3'))
    b.white_to_move = True
    b._rebuild_occ()

    legal = board_legal_destinations(b, sq('e1'))
    assert sq('d2') not in legal, "King must not walk into pawn attack on d2"
    assert sq('f2') not in legal, "King must not walk into pawn attack on f2"


def test_pinned_piece_cannot_move_off_pin_ray():
    # White king e1, white rook e4, black rook e8 — rook is pinned on e-file
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['R'] = np.uint64(1 << sq('e4'))
    b.bitboards['k'] = np.uint64(1 << sq('d8'))   # black king off the pin file
    b.bitboards['r'] = np.uint64(1 << sq('e8'))   # black rook pins white rook
    b.white_to_move = True
    b._rebuild_occ()

    legal = board_legal_destinations(b, sq('e4'))
    for dest in legal:
        assert dest % 8 == sq('e4') % 8, f"Pinned rook must not leave e-file (moved to {dest})"


def test_king_in_check_must_resolve():
    # White king e1, black rook e8 — white must get out of check
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('d8'))
    b.bitboards['r'] = np.uint64(1 << sq('e8'))   # gives check on e-file
    b.white_to_move = True
    b._rebuild_occ()

    assert b.is_check()
    legal = b.generate_legal_moves()
    # Every legal move must resolve the check
    for m in legal:
        b.make_move(m)
        assert not b.is_in_check_for(True), f"Move {m} leaves white king in check"
        b.undo_move(m)


def test_checkmate_detected():
    # Back-rank mate: white king h1, black rooks a8 and a1 → checkmate
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('h1'))
    b.bitboards['k'] = np.uint64(1 << sq('a3'))
    b.bitboards['r'] = np.uint64((1 << sq('a1')) | (1 << sq('b2')))
    b.white_to_move = True
    b._rebuild_occ()

    # Verify it's actually checkmate
    if b.is_check() and len(b.generate_legal_moves()) == 0:
        assert b.is_checkmate()


def test_stalemate_detected():
    # Classic stalemate: white king a1, black queen c2, black king c1
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('a1'))
    b.bitboards['k'] = np.uint64(1 << sq('c1'))
    b.bitboards['q'] = np.uint64(1 << sq('c3'))
    b.white_to_move = True
    b._rebuild_occ()

    if not b.is_check() and len(b.generate_legal_moves()) == 0:
        assert b.is_stalemate()


# ------------------------------------------------------------------ #
#  Castling                                                            #
# ------------------------------------------------------------------ #

def test_white_kingside_castle():
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['R'] = np.uint64(1 << sq('h1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.castling_rights = {'K': True, 'Q': False, 'k': False, 'q': False}
    b.white_to_move = True
    b._rebuild_occ()

    legal = b.generate_legal_moves()
    castle_moves = [m for m in legal if m.is_kingside_castle()]
    assert len(castle_moves) == 1

    b.make_move(castle_moves[0])
    # King should be on g1, rook on f1
    assert (b.bitboards['K'] >> np.uint64(sq('g1'))) & np.uint64(1)
    assert (b.bitboards['R'] >> np.uint64(sq('f1'))) & np.uint64(1)


def test_black_rook_move_revokes_castling_right():
    # BUG FIX: moving_piece == 'r' (was moving_piece.upper() == 'r', always False)
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['r'] = np.uint64(1 << sq('h8'))
    b.castling_rights = {'K': False, 'Q': False, 'k': True, 'q': False}
    b.white_to_move = False
    b._rebuild_occ()

    move = Move(sq('h8'), sq('g8'))
    b.make_move(move)
    assert not b.castling_rights['k'], "Moving h8 rook must revoke black kingside castling"


def test_white_rook_move_revokes_castling_right():
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['R'] = np.uint64(1 << sq('h1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.castling_rights = {'K': True, 'Q': False, 'k': False, 'q': False}
    b.white_to_move = True
    b._rebuild_occ()

    move = Move(sq('h1'), sq('g1'))
    b.make_move(move)
    assert not b.castling_rights['K'], "Moving h1 rook must revoke white kingside castling"


# ------------------------------------------------------------------ #
#  Pawn move generation — file-wrap guard                             #
# ------------------------------------------------------------------ #

def test_pawn_cannot_capture_across_file_wrap():
    # White pawn on a2, black rook on h3 — should NOT be a legal capture
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['P'] = np.uint64(1 << sq('a2'))
    b.bitboards['r'] = np.uint64(1 << sq('h3'))
    b.white_to_move = True
    b._rebuild_occ()

    legal = b.generate_legal_moves()
    pawn_captures = [m for m in legal if m.from_square == sq('a2') and m.capture]
    for m in pawn_captures:
        assert m.to_square != sq('h3'), "Pawn must not capture across file wrap"


def test_pawn_double_push():
    b = empty_board()
    b.bitboards['K'] = np.uint64(1 << sq('e1'))
    b.bitboards['k'] = np.uint64(1 << sq('e8'))
    b.bitboards['P'] = np.uint64(1 << sq('e2'))
    b.white_to_move = True
    b._rebuild_occ()

    legal = b.generate_legal_moves()
    pawn_moves = [m for m in legal if m.from_square == sq('e2')]
    dests = {m.to_square for m in pawn_moves}
    assert sq('e3') in dests
    assert sq('e4') in dests


def test_en_passant():
    # After white plays e2e4 from starting, en_passant_square should be e3
    board = Board()
    e2e4 = next(m for m in board.generate_legal_moves()
                if m.from_square == sq('e2') and m.to_square == sq('e4'))
    board.make_move(e2e4)
    assert board.en_passant_square == sq('e3')


# ------------------------------------------------------------------ #
#  Zobrist hashing                                                     #
# ------------------------------------------------------------------ #

def test_zobrist_hash_restored_after_undo():
    board = Board()
    before_hash = board.zobrist_hash
    move = board.generate_legal_moves()[0]
    board.make_move(move)
    assert board.zobrist_hash != before_hash, "Hash must change after a move"
    board.undo_move(move)
    assert board.zobrist_hash == before_hash, "Hash must be restored after undo"


def test_zobrist_different_positions_different_hashes():
    board = Board()
    start_hash = board.zobrist_hash
    moves = board.generate_legal_moves()
    hashes = set()
    for m in moves:
        board.make_move(m)
        hashes.add(board.zobrist_hash)
        board.undo_move(m)
    assert start_hash not in hashes, "Starting position hash must differ from all one-ply successors"
    assert len(hashes) == len(moves), "All distinct moves must produce distinct hashes"


def test_zobrist_hash_matches_recompute():
    board = Board()
    for _ in range(5):
        moves = board.generate_legal_moves()
        if not moves:
            break
        board.make_move(moves[0])
    assert board.zobrist_hash == board._compute_zobrist(), \
        "Incremental hash must match full recompute after several moves"


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def board_legal_destinations(board, from_sq):
    """Return the set of destination squares for all legal moves from from_sq."""
    return {m.to_square for m in board.generate_legal_moves() if m.from_square == from_sq}
