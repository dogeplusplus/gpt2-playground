
# For some reason there are moves on the 19th position
MOVES = [" "] + [
    (p, (i, j)) for p in ('b', 'w') for i in range(25) for j in range(25)
] + [('w', None), ('b', None), "SOS", "EOS"]

ENCODING = {
    move: i for i, move in enumerate(MOVES)
}

DECODING = {
    i: move for move, i in ENCODING.items()
}


MOVES9 = [" "] + [
    (p, (i, j))
    for p in ('b', 'w')
    for i in range(9)
    for j in range(9)
] + [('w', None), ('b', None), "SOS", "EOS"]

ENCODING9X9 = {
    move: i for i, move in enumerate(MOVES9)
}

DECODING9X9 = {
    v: k for k, v in ENCODING9X9.items()
}
