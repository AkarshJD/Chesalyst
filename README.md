# Chesalyst 

**Chesalyst** is a classical chess engine written in Python.  
It combines bitboard-based move generation, alpha-beta pruning, and heuristic evaluation to simulate thoughtful, human-like play.

---

## Features

- Bitboard-based board representation for fast calculations
- Alpha-beta pruning with iterative deepening
- Static evaluation using piece-square tables and pawn structure
- Legal move generation with check and castling rules
- Custom time controls (5+3, 15+10, 60+30, etc.)
- Undo/redo, draw by repetition, and 50-move rule support
- Saves games as PGN (`last_game.pgn`)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy

### Install dependencies

```bash
pip install numpy
```

### Run the Engine

```bash
python game_main.py
```

Then follow the on-screen instructions to choose time controls and side.

---

## Modes

- **Player vs Engine** – Choose White or Black and play
- **Engine vs Engine** – Watch the machine analyze itself
- **Analysis Mode** – Test positions and get evaluations interactively

---

## Code Structure

```
.
├── board.py        # Bitboard logic and move generation
├── evaluation.py   # Scoring logic for evaluating positions
├── move.py         # Move object representation
├── search.py       # Search algorithms (minimax, alpha-beta)
├── game_main.py    # CLI-based interface and gameplay loop
└── README.md
```

---

## Known Issues

- Backtracking (undo) fails during play vs computer mode
- Engine may get stuck in a loop after a few capture moves
- Evaluation in analysis mode is shallow

---

## License

This project is licensed under the MIT License. Feel free to explore, modify, and build upon it.

---


## Acknowledgements

- Some ideas in the bitboard logic were referenced from the open-source engine **Stockfish** (GPLv3). This project re-implements those concepts in Python for educational purposes.
- Concepts and structural inspiration were drawn from the YouTube video  
  **"The Fascinating Programming of a Chess Engine"** by *Bartek Spitza*.  
  [Watch it here](https://www.youtube.com/watch?v=U4ogK0MIzqk)
- Additional insight and visual understanding were influenced by  
  **"Coding Adventure: Chess"** by *Sebastian Lague*.  
  [Watch it here](https://www.youtube.com/watch?v=U4ogK0MIzqk)  
- Designed for simplicity, clarity, and a desire to learn from scratch.

