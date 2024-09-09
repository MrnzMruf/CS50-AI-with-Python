"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):  # Who is playing?
    """
    Returns player who has the next turn on a board.
    """

    count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != EMPTY:
                count += 1

    if board == initial_state():
        return X
    if count % 2 == 1:
        return O
    else:
        return X


def actions(board):  # new board after playing o and x
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i, j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Action is Invalid")       # try... except...
        # changes to one affect the other --> in borad2 = board --use deepcopy to not affect
    board2 = copy.deepcopy(board)                  # due to the original board does not change  #deepcopy function
    board2[action[0]][action[1]] = player(board)
    return board2


def winner(board):                       # Determine winner
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(len(board)):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            if board[i][0] == 'X':
                return X
            elif board[i][0] == 'O':
                return O
            else:
                return None

    for i in range(len(board)):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            if board[0][i] == 'X':
                return X
            elif board[0][i] == 'O':
                return O
            else:
                return None

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        if board[0][0] == 'X':
            return X
        elif board[0][0] == 'O':
            return O
        else:
            return None

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        if board[0][2] == 'X':
            return X
        elif board[0][2] == 'O':
            return O
        else:
            return None

    return None


def terminal(board):                        # Check the game is done or not (True = Done)
    """
    Returns True if game is over, False otherwise.
    """
    if (winner(board) == X):
        return True
    elif (winner(board) == O):
        return True

    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                return False
    return True


def utility(board):             # if x win == 1 / if o win == -1  / equal = 0
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if (winner(board)) == X:
        return 1
    elif (winner(board)) == O:
        return -1
    else:
        return 0


def minimax(board):                        # the game main!
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    Max = -math.inf     # -binahayat
    Min = float("inf")  # +binahayat

    if player(board) == X:
        return Max_Value(board, Max, Min)[1]
    else:
        return Min_Value(board, Max, Min)[1]


def Max_Value(board, Max, Min):
    move = None
    if terminal(board):
        return [utility(board), None]
    v = -math.inf
    for action in actions(board):
        test = Min_Value(result(board, action), Max, Min)[0]
        Max = max(Max, test)
        if test > v:
            v = test
            move = action
        if Max >= Min:
            break
    return [v, move]


def Min_Value(board, Max, Min):
    move = None
    if terminal(board):
        return [utility(board), None]
    v = float('inf')
    for action in actions(board):
        test = Max_Value(result(board, action), Max, Min)[0]
        Min = min(Min, test)
        if test < v:
            v = test
            move = action
        if Max >= Min:
            break
    return [v, move]
