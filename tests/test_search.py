import numpy as np
import pytest
from chesalyst.board import Board
from chesalyst.evaluation import Evaluator
from chesalyst.search import Searcher


def _board_snapshot(board):
    return {
        'bitboards': {k: int(v) for k, v in board.bitboards.items()},
        'white_to_move': board.white_to_move,
        'castling_rights': board.castling_rights.copy(),
        'en_passant_square': board.en_passant_square,
        'halfmove_clock': board.halfmove_clock,
        'fullmove_number': board.fullmove_number,
        'stack_depth': len(board._state_stack),
    }


def _snapshots_equal(a, b):
    return a == b


# ------------------------------------------------------------------ #
#  Board state preservation (regression for double-undo bug)          #
# ------------------------------------------------------------------ #

def test_search_does_not_corrupt_board():
    """After search(), board state must be exactly as before."""
    board = Board()
    searcher = Searcher(base_time=1, increment=0, depth_limit=2)

    before = _board_snapshot(board)
    move = searcher.search(board, move_number=1)
    after = _board_snapshot(board)

    assert _snapshots_equal(before, after), (
        "Search corrupted board state (double-undo or missing undo)"
    )


def test_search_does_not_corrupt_board_after_several_moves():
    """Play 10 half-moves, run search at each turn, verify state each time."""
    board = Board()
    searcher = Searcher(base_time=2, increment=0, depth_limit=2)

    for i in range(10):
        before = _board_snapshot(board)
        move = searcher.search(board, move_number=i // 2 + 1)

        after = _board_snapshot(board)
        assert _snapshots_equal(before, after), (
            f"Board corrupted after search on move {i + 1}"
        )
        assert len(board._state_stack) == before['stack_depth'], \
            "Search must not change the state stack depth"

        if move is None:
            break
        board.make_move(move)


# ------------------------------------------------------------------ #
#  Search returns a legal move                                         #
# ------------------------------------------------------------------ #

def test_search_returns_legal_move():
    board = Board()
    searcher = Searcher(base_time=60, increment=0, depth_limit=2)
    move = searcher.search(board, move_number=1)

    assert move is not None
    legal = board.generate_legal_moves()
    assert any(m == move for m in legal), "Search returned an illegal move"


def test_search_after_many_moves():
    """Play 20 real moves, search at each, verify a legal move is returned."""
    board = Board()
    searcher = Searcher(base_time=2, increment=0, depth_limit=2)

    for i in range(20):
        move = searcher.search(board, move_number=i // 2 + 1)
        if move is None:
            break
        legal = board.generate_legal_moves()
        assert any(m == move for m in legal), f"Illegal move returned on ply {i + 1}"
        board.make_move(move)
        if board.is_checkmate() or board.is_stalemate():
            break


# ------------------------------------------------------------------ #
#  Time budget not mutated across calls                                #
# ------------------------------------------------------------------ #

def test_time_limit_not_mutated():
    """base_time must stay constant across multiple search() calls."""
    board = Board()
    searcher = Searcher(base_time=300, increment=3, depth_limit=1)
    initial_base = searcher.base_time

    for i in range(5):
        move = searcher.search(board, move_number=i + 1)
        assert searcher.base_time == initial_base, (
            f"base_time changed from {initial_base} to {searcher.base_time} after call {i + 1}"
        )
        if move:
            board.make_move(move)


# ------------------------------------------------------------------ #
#  Mate in one                                                         #
# ------------------------------------------------------------------ #

def test_finds_mate_in_one():
    """Searcher must find the only checkmate move.

    Position: Rc7, Kh1 vs Ka8, pa7, pb7 (white to move).
    The only mate in one is Rc7-c8#:
      - Rc8 checks black king via the 8th rank
      - a7 and b7 are blocked by black's own pawns
      - the king cannot capture the rook (c8 is 2 files away)
    """
    board = Board()
    board.bitboards = {k: np.uint64(0) for k in board.bitboards}

    def sq(name):
        files = 'abcdefgh'
        f = files.index(name[0])
        r = int(name[1]) - 1
        return (7 - r) * 8 + f

    board.bitboards['R'] = np.uint64(1 << sq('c7'))
    board.bitboards['K'] = np.uint64(1 << sq('h1'))
    board.bitboards['k'] = np.uint64(1 << sq('a8'))
    board.bitboards['p'] = np.uint64((1 << sq('a7')) | (1 << sq('b7')))
    board.white_to_move = True
    board.castling_rights = {'K': False, 'Q': False, 'k': False, 'q': False}

    searcher = Searcher(base_time=60, depth_limit=3)
    move = searcher.search(board, move_number=1)

    assert move is not None
    board.make_move(move)
    assert board.is_checkmate(), f"Expected checkmate after {move}, but board is not in checkmate"
