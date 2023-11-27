import numpy as np

def get_actions(state):
    '''
    Returns a list of possible actions (columns to place a piece in)
    :param state: 6x7 matrix
    :return: list of ints
    '''
    return [col for col in range(7) if state[-1, col] == 0]

def apply_action(state, action, player):
    # Apply the action to the given state
    row = min([r for r in range(6) if state[r, action] == 0])
    state[row, action] = player
    return state

def is_terminal(state):
    '''
    function to find out if the game is over and if so what is the reward
    :param state: 6x7 matrix
    :return: boolean, int
    '''

    # Check for a win
    for player in [1, -1]:
        # loop over all rows
        for row in range(6):
            # win in a row
            for col in range(4):
                if np.all(state[row, col:col+4] == player):
                    return True, player
            # win in a column
            if row <= 2:
                for col in range(7):
                    if np.all(state[row:row+4, col] == player):
                        return True, player
                # win on first diagonal
                for col in range(4):
                    if np.all(state[row:row+4, col:col+4].diagonal() == player):
                        return True, player
                # win on second diagonal
                for col in range(3, 7):
                    if np.all(np.fliplr(state[row:row+4, col-3:col+1]).diagonal() == player):
                        return True, player

    # Check for a draw
    if np.all(state != 0):
        return True, 0

    return False, None

