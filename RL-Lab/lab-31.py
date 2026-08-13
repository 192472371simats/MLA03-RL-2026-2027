import random
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2
def get_actions(board):
    return [
        i for i in range(9)
        if board[i] == 0
    ]
def choose_action(board):
    state = tuple(board)
    actions = get_actions(board)
    if state not in Q:
        Q[state] = {a: 0.0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)
    return max(
        actions,
        key=lambda a: Q[state][a]
    )
def winner(board):
    lines = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a, b, c in lines:
        if (
            board[a] != 0 and
            board[a] == board[b] == board[c]
        ):
            return board[a]
    if 0 not in board:
        return 0
    return None
for episode in range(5000):
    board = [0] * 9
    state = tuple(board)
    while True:
        action = choose_action(board)
        board[action] = 1
        result = winner(board)
        if result is not None:
            reward = 1 if result == 1 else 0
            Q[state][action] += alpha * (
                reward - Q[state][action]
            )
            break
        next_state = tuple(board)
        next_action = choose_action(board)
        next_value = Q[next_state][next_action]
        reward = 0
        Q[state][action] += alpha * (
            reward
            + gamma * next_value
            - Q[state][action]
        )
        state = next_state
        board[next_action] = -1
        result = winner(board)
        if result is not None:
            reward = -1 if result == -1 else 0
            Q[state][next_action] += alpha * (
                reward - Q[state][next_action]
            )
            break
print("SARSA Training Completed")
print("Number of Learned States:", len(Q))