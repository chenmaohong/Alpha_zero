import numpy as np


class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = (1, 2)
        self.states = {}

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.available = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        0 1 2
        3 4 5
        6 7 8
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        else:
            h = location[0]
            w = location[1]
            move = h * self.width + w
            if move not in range(self.height * self.width):
                return -1
        return move

    def do_move(self, move):
        self.states[move] = self.current_player
        self.available.remove(move)
        self.current_player = 3 - self.current_player
        self.last_move = move

    def get_current_player(self):
        return self.current_player

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
            if len(self.states) % 2 == 0:
                square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        moved = list(set(range(width * height)) - set(self.available))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if w in range(width - n + 1):
                flag = 1
                for i in range(m, m + n):
                    if states.get(i, -1) != player:
                        flag = 0
                        break
                if flag:
                    return True, player
            if h in range(height - n + 1):
                flag = 1
                for i in range(m, m + n * width, width):
                    if states.get(i, -1) != player:
                        flag = 0
                        break
                if flag:
                    return True, player
            if w in range(width - n + 1) and h in range(height - n + 1):
                flag = 1
                for i in range(m, m + n * width + n, width + 1):
                    if states.get(i, -1) != player:
                        flag = 0
                        break
                if flag:
                    return True, player

            if w in range(n - 1, width) and h in range(height - n + 1):
                flag = 1
                for i in range(m, m + n * width - n, width - 1):
                    if states.get(i, -1) != player:
                        flag = 0
                        break
                if flag:
                    return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.available) == 0:
            return True, -1
        return False, -1


class Game(object):
    def __init__(self, board):
        self.board = board

    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while (1):
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            #保存self-play数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end.Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 or 1')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while (1):
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
