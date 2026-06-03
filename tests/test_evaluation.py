import numpy as np
import pytest
from chesalyst.board import Board
from chesalyst.evaluation import Evaluator


def test_starting_position_near_zero():
    board = Board()
    ev = Evaluator()
    score = ev.evaluate(board)
    # Starting position should be close to symmetric (small tempo bonus)
    assert abs(score) < 50


def test_material_imbalance():
    board = Board()
    ev = Evaluator()

    # Remove a white queen
    board.bitboards['Q'] = np.uint64(0)
    score = ev.evaluate(board)
    assert score < -800, "Losing a queen should give a strongly negative score for white"


def test_extra_pawn_favors_white():
    board = Board()
    ev = Evaluator()

    # Remove one black pawn
    board.bitboards['p'] = np.uint64(board.bitboards['p'] & ~np.uint64(1 << 8))
    score = ev.evaluate(board)
    assert score > 50, "One extra pawn should favor white"


def test_evaluate_empty_board_kings_only():
    board = Board()
    ev = Evaluator()
    board.bitboards = {k: np.uint64(0) for k in board.bitboards}
    board.bitboards['K'] = np.uint64(1 << 60)
    board.bitboards['k'] = np.uint64(1 << 4)
    score = ev.evaluate(board)
    # Kings cancel out; should be close to zero
    assert abs(score) < 100
