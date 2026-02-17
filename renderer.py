import pygame
import sys
from environment import Environment

class PygameRenderer:
    def __init__(self, env: Environment, cell_size: int = 80, padding: int = 10):
        pygame.init()
        self.env = env
        self.cell = cell_size
        self.padding = padding
        w = env.cols * cell_size
        h = env.rows * cell_size
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Connect Four")
        self.clock = pygame.time.Clock()

    def draw(self):
        self.screen.fill((30, 30, 60))  # background
        # draw cells, row 0 is bottom -> translate
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x = c * self.cell + self.cell // 2
                # draw circle center
                y = (self.env.rows - 1 - r) * self.cell + self.cell // 2
                val = self.env.board[r, c]
                if val == 0:
                    color = (200, 200, 200)
                elif val == 1:
                    color = (220, 60, 60)   # player1 red-ish
                else:
                    color = (60, 220, 60)   # player2 green-ish
                pygame.draw.circle(self.screen, color, (x, y), int(self.cell * 0.4))
        pygame.display.flip()

    def handle_events(self):
        """Handle quit events and return None or last clicked column index."""
        clicked_col = None
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                col = mx // self.cell
                if 0 <= col < self.env.cols:
                    clicked_col = col
        return clicked_col

    def play_human_vs_human(self):
        """Simple loop to play locally with two humans clicking columns."""
        while True:
            self.clock.tick(30)
            click = self.handle_events()
            if click is not None:
                try:
                    self.env.make_move(self.env.current_player, click)
                except ValueError as e:
                    print(e)
            self.draw()
            if self.env.is_terminal():
                winner = self.env.winner_name()
                print("Game over. Winner:", winner)
                pygame.time.wait(2000)
                self.env.reset()

if __name__ == "__main__":
    from environment import Environment
    env = Environment()
    renderer = PygameRenderer(env)
    renderer.play_human_vs_human()