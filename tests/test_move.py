from chesalyst.move import Move


def test_str_quiet():
    m = Move(52, 36)  # e2e4
    assert str(m) == 'e2e4'


def test_str_promotion():
    m = Move(8, 0, promotion='Q')  # e7e8=Q
    assert '=Q' in str(m)


def test_str_capture():
    m = Move(52, 36, capture=True)
    assert str(m) == 'e2e4'


def test_is_pawn_move():
    assert Move(52, 36).is_pawn_move()   # double push
    assert Move(52, 44).is_pawn_move()   # single push
    assert Move(36, 27).is_pawn_move()   # diagonal capture (offset 9)
    assert not Move(57, 42).is_pawn_move()  # knight jump (offset 15)


def test_is_capture():
    assert Move(0, 1, capture=True).is_capture()
    assert Move(0, 1, en_passant=True).is_capture()
    assert not Move(0, 1).is_capture()


def test_castle_flags():
    m = Move(60, 62, castle='kingside')
    assert m.is_kingside_castle()
    assert not m.is_queenside_castle()

    m2 = Move(60, 58, castle='queenside')
    assert m2.is_queenside_castle()
    assert not m2.is_kingside_castle()


def test_equality():
    a = Move(52, 36)
    b = Move(52, 36)
    c = Move(52, 28)
    assert a == b
    assert a != c
    assert a != "e2e4"
