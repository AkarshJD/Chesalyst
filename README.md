# Chesalyst

**Chesalyst** is a classical chess engine written in Python.
It combines bitboard-based move generation, alpha-beta pruning, and heuristic evaluation to play legal, thoughtful chess.

---

## Features

- Bitboard-based board representation
- Alpha-beta pruning with iterative deepening
- Static evaluation using piece-square tables and pawn structure (doubled, isolated, passed pawns)
- Full legal move generation: check filtering, pins, castling, en passant, promotion
- Custom time controls (5+3, 15+10, 60+30, etc.)
- Draw detection: threefold repetition, 50-move rule, insufficient material, stalemate
- Saves games as PGN (`last_game.pgn`)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- NumPy

### Install

```bash
pip install -e .
```

Or without installing:

```bash
pip install numpy
python main.py
```

### Run

```bash
python main.py
```

Follow the prompts to choose a time control and side.

---

## Move Input

Moves are entered in long algebraic notation:

| Input | Meaning |
|-------|---------|
| `e2e4` | Move piece from e2 to e4 |
| `g1f3` | Knight to f3 |
| `e7e8q` | Pawn promotes to queen |
| `O-O` | Kingside castle |
| `O-O-O` | Queenside castle |
| `engine` | Let the engine play your move |
| `draw` | Offer a draw |
| `resign` | Resign |
| `exit` | Quit |

---

## Modes

- **Player vs Engine** — choose White or Black
- **Analysis Mode** — type `engine` each turn to watch the engine play both sides

---

## Project Structure

```
.
├── src/
│   └── chesalyst/
│       ├── __init__.py
│       ├── board.py        # Bitboard representation and legal move generation
│       ├── evaluation.py   # Material, piece-square tables, pawn structure
│       ├── move.py         # Move object
│       └── search.py       # Iterative deepening alpha-beta
├── tests/
│   ├── test_board.py
│   ├── test_evaluation.py
│   ├── test_move.py
│   └── test_search.py
├── main.py                 # CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## License

MIT License. Feel free to explore, modify, and build upon it.

---

## Acknowledgements

- Bitboard concepts referenced from **Stockfish** (GPLv3), re-implemented in Python for educational purposes.
- Structural inspiration from **"The Fascinating Programming of a Chess Engine"** by Bartek Spitza.
- Visual intuition from **"Coding Adventure: Chess"** by Sebastian Lague.
