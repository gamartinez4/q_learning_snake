#main.py

import pygame
import sys
import numpy as np
from snake_game import SnakeGame, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, BLACK, WHITE, GREEN, RED, BLUE # Added BLUE if used for head
from rl_agent import QLearningAgent

# --- Configuration ---
TRAIN_FPS = 60      # Faster for training (if not rendering much)
PLAY_FPS = 10       # Normal speed for playing/testing
RENDER_EVERY_N_EPISODES = 10 # Render 1 out of N episodes during training
SAVE_EVERY_N_EPISODES = 100 # Save Q-Table every N episodes

GREY = (150, 150, 150)

# --- Pygame Initialization ---
pygame.init()
# Added space for improved UI
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
pygame.display.set_caption("Snake RL - Q-Learning")
clock = pygame.time.Clock()

# Use available fonts or specify path to a .ttf
try:
    font = pygame.font.SysFont('Arial', 36)
    small_font = pygame.font.SysFont('Arial', 24)
    ui_font = pygame.font.SysFont('Consolas', 18) # Monospaced font for UI
except pygame.error:
    print("Arial/Consolas font not found, using default font.")
    font = pygame.font.SysFont(None, 40) # Fallback
    small_font = pygame.font.SysFont(None, 30)
    ui_font = pygame.font.SysFont(None, 22)


def draw_text(text, font, color, surface, x, y, center=False, topleft=False, topright=False):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    if center:
        textrect.center = (x, y)
    elif topleft:
        textrect.topleft = (x, y)
    elif topright:
        textrect.topright = (x, y)
    else: # Default to topleft
        textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def main_menu():
    selected_option = 0 # 0: Learn, 1: Test
    options = ["Learn", "Test"]
    pointer = "> "
    no_pointer = "  "

    while True:
        screen.fill(BLACK)
        draw_text("Snake RL - Q-Learning", font, GREEN, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 5, center=True)

        for i, option_text in enumerate(options):
            y_pos = SCREEN_HEIGHT // 2 + i * 60
            display_text = ""
            if i == selected_option:
                display_text = pointer + option_text
                color = WHITE
            else:
                display_text = no_pointer + option_text
                color = GREY

            draw_text(display_text, small_font, color, screen, SCREEN_WIDTH // 2, y_pos, center=True)

        draw_text("Use UP/DOWN to select, ENTER to confirm", ui_font, GREY, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.85, center=True)
        draw_text("[ESC] Exit", ui_font, GREY, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.9, center=True)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1 + len(options)) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected_option].lower()
                elif event.key == pygame.K_ESCAPE:
                      pygame.quit()
                      sys.exit()


def game_loop(mode):
    game = SnakeGame()

    # Get state size from environment
    initial_state_example = game.reset()
    if not isinstance(initial_state_example, tuple): # QLearningAgent expects tuples as keys
         raise TypeError(f"game.reset() must return a tuple state, but returned {type(initial_state_example)}")
    state_size = len(initial_state_example)

    # Agent takes 3 relative actions: 0 (straight), 1 (left), 2 (right)
    action_size = 3

    # Create agent. Q-Table is loaded (or created) INSIDE __init__
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    # ---- NO NEED TO CALL agent.load_model() ----
    # Loading happens automatically when creating the QLearningAgent object above.

    running = True
    episode = 0
    max_score_ever = 0
    scores_window = [] # For recent average calculation

    current_fps = TRAIN_FPS if mode == "learn" else PLAY_FPS

    while running:
        if mode == "learn":
            # --- Training Loop ---
            state = game.reset() # Ensures it returns a tuple
            done = False
            total_reward_episode = 0
            episode_steps = 0

            # Control whether to render this episode
            render_this_episode = (episode % RENDER_EVERY_N_EPISODES == 0)

            while not done:
                # Process events to keep window responsive and allow exit/save
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save_q_table()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            agent.save_q_table()
                            print("Training interrupted. Q-Table saved.")
                            return # Return to menu
                        if event.key == pygame.K_s:
                            agent.save_q_table()
                            print("Q-Table manually saved.")

                # --- Agent Logic ---
                # Choose action using epsilon-greedy (exploration=True by default in learn)
                action = agent.choose_action(state, training=True)

                # Execute action in environment
                next_state, reward, done, score = game.step(action)

                # Learn from experience
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward_episode += reward
                episode_steps += 1

                # Conditional rendering
                if render_this_episode:
                    screen.fill(BLACK)
                    game.render(screen)
                    # UI during training
                    draw_text(f"MODE: LEARNING [Episode {episode + 1}]", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 5, topleft=True)
                    draw_text(f"Score: {score}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 25, topleft=True)
                    draw_text(f"Max Score: {max_score_ever}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 45, topleft=True)
                    draw_text(f"Steps: {episode_steps}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 65, topleft=True)

                    avg_score = np.mean(scores_window[-100:]) if scores_window else 0.0
                    draw_text(f"Avg Score(100): {avg_score:.2f}", ui_font, WHITE, screen, SCREEN_WIDTH - 5, SCREEN_HEIGHT + 5, topright=True)
                    draw_text(f"Epsilon: {agent.epsilon:.4f}", ui_font, WHITE, screen, SCREEN_WIDTH - 5, SCREEN_HEIGHT + 25, topright=True)
                    draw_text(f"Q-Size: {len(agent.q_table)}", ui_font, WHITE, screen, SCREEN_WIDTH - 5, SCREEN_HEIGHT + 45, topright=True)
                    draw_text(f"[ESC] Menu [S] Save", ui_font, GREY, screen, SCREEN_WIDTH - 5, SCREEN_HEIGHT + 65, topright=True)

                    pygame.display.flip()
                    clock.tick(current_fps) # Control FPS only if rendering
                # Allow Pygame to process events even if not rendering to prevent hanging
                elif episode_steps % 100 == 0:
                    pygame.event.pump()


            # --- End of Episode (Learning) ---
            episode += 1
            scores_window.append(score)
            max_score_ever = max(max_score_ever, score)

            # Print progress every N episodes
            if episode % 20 == 0: # Print every 20 episodes
                avg_s = np.mean(scores_window[-100:])
                print(f"Episode {episode} | Score: {score} | Avg Score(100): {avg_s:.2f} | Epsilon: {agent.epsilon:.4f} | Steps: {episode_steps} | Q-Size: {len(agent.q_table)}")

            # Save Q-table periodically
            if episode % SAVE_EVERY_N_EPISODES == 0:
                agent.save_q_table()

        elif mode == "test":
            # --- Testing Loop ---
            state = game.reset()
            done = False
            total_reward_episode = 0
            episode_steps = 0
            agent.epsilon = 0 # ENSURE NO EXPLORATION DURING TESTING

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return # Return to menu

                # --- Agent Logic (Exploitation Only) ---
                # Choose best known action (training=False)
                action = agent.choose_action(state, training=False)

                # Execute action
                next_state, reward, done, score = game.step(action)
                # DO NOT CALL agent.learn() in test mode

                state = next_state
                total_reward_episode += reward
                episode_steps += 1

                # Always render in test mode
                screen.fill(BLACK)
                game.render(screen)
                # UI during testing
                # Try to get max_score from agent if exists, otherwise N/A
                try:
                    # Check if there's any saved score (indirectly via q_table)
                    max_score_display = max_score_ever if max_score_ever > 0 else 'N/A'
                    if max_score_ever == 0 and len(agent.q_table) > 0:
                        max_score_display = '(Trained, max score not recorded)' # Indicates table exists but no max score
                except AttributeError: # If max_score_ever wasn't defined
                    max_score_display = 'N/A'

                draw_text(f"MODE: TESTING (AI)", ui_font, GREEN, screen, 5, SCREEN_HEIGHT + 5, topleft=True)
                draw_text(f"Score: {score}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 25, topleft=True)
                draw_text(f"AI Max Score: {max_score_display}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 45, topleft=True)
                draw_text(f"Steps: {episode_steps}", ui_font, WHITE, screen, 5, SCREEN_HEIGHT + 65, topleft=True)
                draw_text(f"[ESC] Return to Menu", ui_font, GREY, screen, SCREEN_WIDTH - 5, SCREEN_HEIGHT + 65, topright=True)

                pygame.display.flip()
                clock.tick(current_fps)

            # --- End of Game (Testing) ---
            print(f"Test Game Ended. Score: {game.score}, Steps: {episode_steps}")
            draw_text("Game Over!", font, RED, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20, center=True)
            draw_text(f"Final Score: {game.score}", small_font, WHITE, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30, center=True)

            pygame.display.flip()
            pygame.time.wait(3000) # Pause before returning to menu
            return # Automatically return to menu


if __name__ == "__main__":
    while True:
        try:
            selected_mode = main_menu()
            if selected_mode in ["learn", "test"]:
                 game_loop(selected_mode)
            else:
                 print("Invalid mode selected or menu closed.")
                 break # Exit if menu doesn't return a valid option
        except KeyboardInterrupt: # Handle Ctrl+C cleanly
             print("\nInterruption detected (Ctrl+C). Exiting...")
             # Consider saving model here too if in the middle of training
             # (Though there's periodic saving and saving when exiting with ESC)
             pygame.quit()
             sys.exit()
        except Exception as e: # Catch other unexpected errors
             print(f"\nUNEXPECTED ERROR!: {e}")
             import traceback
             traceback.print_exc() # Print full error details
             pygame.quit()
             sys.exit()
