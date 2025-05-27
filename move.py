class Move:
    def __init__(self, from_square, to_square, promotion=None, capture=False, castle=None, en_passant=False):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
        self.capture = capture
        self.castle = castle  # 'kingside', 'queenside', or None
        self.en_passant = en_passant

    def is_kingside_castle(self):
        return self.castle == 'kingside'

    def is_queenside_castle(self):
        return self.castle == 'queenside'

    def is_pawn_move(self):
        """
        Detect a pawn move by distance:
        ‑ quiet: 8 or 16 squares (straight ahead)
        ‑ capture / en‑passant: 7 or 9 squares
        These offsets never occur for Knights, Bishops, etc.
        """
        delta = abs(self.to_square - self.from_square)
        return delta in (7, 8, 9, 16)

    def is_capture(self):
        return self.capture or self.en_passant

    def __str__(self):
        files = 'abcdefgh'
        ranks = '12345678'
        from_file = files[self.from_square % 8]
        from_rank = ranks[7 - self.from_square // 8]
        to_file = files[self.to_square % 8]
        to_rank = ranks[7 - self.to_square // 8]
        move_str = f"{from_file}{from_rank}{to_file}{to_rank}"
        if self.promotion:
            move_str += f"={self.promotion.upper()}"
        return move_str


