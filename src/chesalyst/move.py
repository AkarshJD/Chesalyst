class Move:
    def __init__(self, from_square, to_square, promotion=None, capture=False, castle=None, en_passant=False):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
        self.capture = capture
        self.castle = castle
        self.en_passant = en_passant

    def is_kingside_castle(self):
        return self.castle == 'kingside'

    def is_queenside_castle(self):
        return self.castle == 'queenside'

    def is_pawn_move(self):
        delta = abs(self.to_square - self.from_square)
        return delta in (7, 8, 9, 16)

    def is_capture(self):
        return self.capture or self.en_passant

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and
                self.to_square == other.to_square and
                self.promotion == other.promotion and
                self.castle == other.castle and
                self.en_passant == other.en_passant)

    def __repr__(self):
        return str(self)

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
