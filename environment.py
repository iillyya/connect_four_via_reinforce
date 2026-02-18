import numpy as np
from typing import List, Optional, Tuple, Dict, Any

class Environment:
    """
    Connect-Four style environment (numpy-only core).

    Board coordinates:
      - self.board is shape (rows, cols)
      - row 0 is the BOTTOM row (so pieces "drop" to increasing row index)
      - values: 0 = empty, 1 = player1, -1 = player2

    Methods:
      - reset() -> observation
      - step(action) -> (obs, reward, done, info)
      - available_actions() -> list of valid column indices
      - render(...) -> external renderer should be used (see separate PygameRenderer)
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        win_length: int = 4,
        player1: str = "player1",
        player2: str = "player2",
        who_starts: str = None,
        agent: Optional[str] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.player1 = player1
        self.player2 = player2
        self.agent = agent or player1  # which player the RL agent corresponds to (for rewards)
        # board with row 0 at bottom
        self.board = np.zeros((rows, cols), dtype=int)
        if who_starts is None:
            who_starts = player1
        if who_starts not in [player1, player2]:
            raise ValueError(f"who_starts must be either {player1} or {player2}")
        self.current_player = who_starts
        self.history = [self.board.copy()]
        self.filled_positions = [0] * self.cols  # how many pieces in each column
        self._winner_cached: Optional[int] = None  # 1, -1, or None

    def reset(self) -> np.ndarray:
        """Reset board to empty. Returns initial observation."""
        self.board.fill(0)
        self.history = [self.board.copy()]
        self.filled_positions = [0] * self.cols
        self._winner_cached = None
        return self.get_observation()

    def available_actions(self) -> List[int]:
        """Return list of columns that are not full."""
        return [c for c in range(self.cols) if self.filled_positions[c] < self.rows]

    def _apply_move_marker(self, marker: int, position: int):
        """Place marker (1 or -1) in column `position`. Assumes move is valid."""
        row = self.filled_positions[position]
        self.board[row, position] = marker
        self.filled_positions[position] += 1
        self.history.append(self.board.copy())
        self._winner_cached = None

    def make_move(self, whose_turn: str, position: int):
        if whose_turn not in [self.player1, self.player2]:
            raise ValueError(f"whose_turn must be either {self.player1} or {self.player2}")
        if position < 0 or position >= self.cols:
            raise ValueError(f"invalid move: position must be between 0 and {self.cols - 1}")
        if self.filled_positions[position] >= self.rows:
            raise ValueError(f"invalid move: column {position} is full")

        marker = 1 if whose_turn == self.player1 else -1
        self._apply_move_marker(marker, position)

        winner = self.get_winner()
        if winner is not None:
            winner_name = self.player1 if winner == 1 else self.player2
            print(f"Player {whose_turn} ({winner_name}) won")

        # swap current player
        self.current_player = self.player2 if whose_turn == self.player1 else self.player1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply action for self.current_player.
        Returns (observation, reward, done, info).
        Reward is from the perspective of `self.agent`:
          - +1.0 if agent just won,
          - -1.0 if agent just lost,
          - 0.0 otherwise (including non-terminal and draws).
        """
        if action not in self.available_actions():
            raise ValueError(f"Action {action} is invalid or column is full.")

        marker = 1 if self.current_player == self.player1 else -1
        self._apply_move_marker(marker, action)

        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        winner_marker = self.get_winner()
        if winner_marker is not None:
            done = True
            winner_name = self.player1 if winner_marker == 1 else self.player2
            info["winner"] = winner_name
            # reward from agent perspective
            if self.agent == winner_name:
                reward = 1.0
            else:
                reward = -1.0
        elif all(pos >= self.rows for pos in self.filled_positions):
            # draw
            done = True
            info["winner"] = None
            reward = 0.0

        # swap current player for next turn
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1

        return self.get_observation(), reward, done, info

    def is_terminal(self) -> bool:
        """True if there is a winner or draw (board full)."""
        if self.get_winner() is not None:
            return True
        if all(pos >= self.rows for pos in self.filled_positions):
            return True
        return False

    def get_winner(self) -> Optional[int]:
        """
        Scan the board for a winning run.
        Returns:
          - 1 if player1 (marker 1) has a win
          - -1 if player2 (marker -1) has a win
          - None if no winner
        This method sets/uses an internal cache to avoid repeated scans.
        """
        if self._winner_cached is not None:
            return self._winner_cached

        b = self.board
        R, C = self.rows, self.cols
        L = self.win_length

        def check_run(r, c, dr, dc):
            start = b[r, c]
            if start == 0:
                return None
            for i in range(1, L):
                rr = r + dr * i
                cc = c + dc * i
                if rr < 0 or rr >= R or cc < 0 or cc >= C or b[rr, cc] != start:
                    return None
            return start

        # scan all cells
        for r in range(R):
            for c in range(C):
                if b[r, c] == 0:
                    continue
                # vertical (dr=1, dc=0)
                if r + (L - 1) < R:
                    res = check_run(r, c, 1, 0)
                    if res is not None:
                        self._winner_cached = int(res)
                        return self._winner_cached
                # horizontal (dr=0, dc=1)
                if c + (L - 1) < C:
                    res = check_run(r, c, 0, 1)
                    if res is not None:
                        self._winner_cached = int(res)
                        return self._winner_cached
                # diagonal down-right (dr=1, dc=1)
                if r + (L - 1) < R and c + (L - 1) < C:
                    res = check_run(r, c, 1, 1)
                    if res is not None:
                        self._winner_cached = int(res)
                        return self._winner_cached
                # diagonal down-left (dr=1, dc=-1)
                if r + (L - 1) < R and c - (L - 1) >= 0:
                    res = check_run(r, c, 1, -1)
                    if res is not None:
                        self._winner_cached = int(res)
                        return self._winner_cached

        self._winner_cached = None
        return None

    def winner_name(self) -> Optional[str]:
        """Return player name (player1/player2) or None."""
        w = self.get_winner()
        if w == 1:
            return self.player1
        if w == -1:
            return self.player2
        return None

    def get_observation(self) -> np.ndarray:
        """Return a copy of the board (safe for the caller to mutate)."""
        return self.board.copy()

    def print_board(self):
        """Print board to console with top row first."""
        symbol = {0: ".", 1: "X", -1: "O"}
        # print top row first
        for r in range(self.rows - 1, -1, -1):
            row_str = " ".join(symbol[int(self.board[r, c])] for c in range(self.cols))
            print(row_str)
        print("-" * (2 * self.cols - 1))
        print(" ".join(str(c) for c in range(self.cols)))
        