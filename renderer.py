import pygame
import sys
from environment import Environment
from argparse import ArgumentParser
import torch
import os

from train import PolicyNet
from train import encode_state, masked_action_distribution
from train import load_policy  



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


    def play_human_vs_model(self, model_path: str, who_starts: str = "alternate"):
        policy = load_policy(model_path)
        policy.eval()

        game_count = 0

        def configure_starting_player():
            nonlocal game_count

            if who_starts == "human":
                self.env.current_player = self.env.player1
            elif who_starts == "model":
                self.env.current_player = self.env.player2
            else:  # alternate
                if game_count % 2 == 0:
                    self.env.current_player = self.env.player1
                else:
                    self.env.current_player = self.env.player2

            game_count += 1

        self.env.reset()
        configure_starting_player()

        model_needs_to_move = True

        while True:
            self.clock.tick(30)
            click = self.handle_events()

            human_player = self.env.player1
            model_player = self.env.player2

            if self.env.current_player == human_player:
                model_needs_to_move = True

                if click is not None:
                    try:
                        self.env.step(click)
                    except ValueError as e:
                        print(e)

            else:
                if model_needs_to_move:
                    pygame.time.wait(300)

                    state = encode_state(self.env.board)
                    with torch.no_grad():
                        logits = policy(state.unsqueeze(0)).squeeze(0)
                        valid_actions = self.env.available_actions()
                        dist = masked_action_distribution(logits, valid_actions)
                        action = dist.probs.argmax().item()

                    self.env.step(action)
                    model_needs_to_move = False

            self.draw()

            if self.env.is_terminal():
                winner = self.env.winner_name()
                print("Game over. Winner:", winner)
                pygame.time.wait(1500)

                self.env.reset()
                configure_starting_player()
                model_needs_to_move = True

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--model_path", type=str, default="connect_4_bot.pt")
    argparser.add_argument("--who_starts", type=str, choices=["human", "model", "alternate"], default="alternate")
    args = argparser.parse_args()

    from environment import Environment
    env = Environment(player1="player1", player2="player2", who_starts="player1", agent="player2")
    renderer = PygameRenderer(env)
    renderer.play_human_vs_model(args.model_path, who_starts=args.who_starts)