#snake_game.py

import pygame
import random
from collections import namedtuple

# --- Game Constants ---
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Snake body
RED = (255, 0, 0)    # Food
BLUE = (0, 0, 255)   # Head

# Directions (as vectors and constants)
Point = namedtuple('Point', 'x y')
UP = Point(0, -1)
DOWN = Point(0, 1)
LEFT = Point(-1, 0)
RIGHT = Point(1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]  # U=0, R=1, D=2, L=3 (for turn indexing)

# Rewards
REWARD_FOOD = 10.0
REWARD_DEATH = -100.0
REWARD_STEP = -0.1    # Small penalty per step to encourage efficiency
REWARD_HUNGER = -50.0  # Penalty for moving too long without eating

# Step limit without food to prevent infinite loops
MAX_STEPS_WITHOUT_FOOD_FACTOR = 1.5  # Multiplier for total grid area

class SnakeGame:
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.max_steps_without_food = int(self.width * self.height * MAX_STEPS_WITHOUT_FOOD_FACTOR)
        # Pygame not initialized here - done in main.py
        self.reset()

    def reset(self):
        """Resets game state for a new episode."""
        # Centered initial position
        start_x, start_y = self.width // 2, self.height // 2
        self.head = Point(start_x, start_y)
        # Initial body (head only)
        self.snake = [self.head]
        # Random initial direction
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False
        self.steps_since_last_food = 0
        # Return initial state as tuple of integers
        return self._get_state()

    def _place_food(self):
        """Places food at random position not occupied by the snake."""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break  # Food successfully placed

    def _get_relative_coords(self, point, head_pos, direction):
        """Converts absolute coordinates to head- and direction-relative coordinates."""
        dx_abs, dy_abs = point.x - head_pos.x, point.y - head_pos.y
        dir_x, dir_y = direction.x, direction.y

        # Rotation based on current direction to align relative Y axis with forward direction
        if direction == UP:      # (0, -1) -> No rotation
            rel_x, rel_y = dx_abs, dy_abs
        elif direction == DOWN:  # (0, 1) -> 180 degree rotation
            rel_x, rel_y = -dx_abs, -dy_abs
        elif direction == LEFT:  # (-1, 0) -> 90 degree clockwise rotation
            rel_x, rel_y = -dy_abs, dx_abs
        elif direction == RIGHT: # (1, 0) -> 90 degree counter-clockwise rotation
            rel_x, rel_y = dy_abs, -dx_abs
        else:  # Fallback (shouldn't occur)
            rel_x, rel_y = dx_abs, dy_abs

        return rel_x, rel_y

    def _get_state(self):
        """
        Calculates current game state for RL agent.
        State includes immediate dangers and food's relative direction.
        Returns: Tuple of integers (0 or 1) representing state.
        """
        head = self.snake[0]

        # --- Calculate relative adjacent points ---
        # Get current direction index
        current_dir_idx = DIRECTIONS.index(self.direction)

        # Straight direction (no change)
        dir_s = self.direction
        # Left turn direction (-90 degrees)
        dir_l = DIRECTIONS[(current_dir_idx - 1 + 4) % 4]
        # Right turn direction (+90 degrees)
        dir_r = DIRECTIONS[(current_dir_idx + 1) % 4]

        # Corresponding points
        point_s = Point(head.x + dir_s.x, head.y + dir_s.y)
        point_l = Point(head.x + dir_l.x, head.y + dir_l.y)
        point_r = Point(head.x + dir_r.x, head.y + dir_r.y)

        # --- Evaluate dangers ---
        # 1 if danger (wall or body), 0 otherwise
        danger_straight = int(self._is_collision(point_s))
        danger_left = int(self._is_collision(point_l))
        danger_right = int(self._is_collision(point_r))

        # --- Evaluate food direction ---
        # Use head-relative coordinates
        food_rel_x, food_rel_y = self._get_relative_coords(self.food, head, self.direction)

        # 1 if food is in that relative direction, 0 otherwise
        food_left = int(food_rel_x < 0)
        food_right = int(food_rel_x > 0)
        food_ahead = int(food_rel_y < 0)  # 'Forward' relative is negative Y

        # --- Build state tuple ---
        # State: (danger_straight, danger_left, danger_right, food_left, food_right, food_ahead)
        state = (
            danger_straight,
            danger_left,
            danger_right,
            food_left,
            food_right,
            food_ahead,
        )
        return tuple(state)

    def _is_collision(self, point=None):
        """Checks if given point collides with walls or snake body."""
        if point is None:
            point = self.head  # Default to current head position

        # Wall collision
        if not (0 <= point.x < self.width and 0 <= point.y < self.height):
            return True

        # Self-collision (excluding head if checking head)
        if point in self.snake[1:]:
            return True

        return False

    def step(self, action):
        """
        Executes game step based on agent's action.
        Args:
            action (int): 0 (straight), 1 (left turn), 2 (right turn)
        Returns:
            tuple: (next_state, reward, game_over, score)
        """
        self.steps_since_last_food += 1
        reward = REWARD_STEP  # Base movement penalty

        # --- Determine new direction ---
        current_dir_idx = DIRECTIONS.index(self.direction)
        new_dir_idx = current_dir_idx

        if action == 1:  # Left turn
            new_dir_idx = (current_dir_idx - 1 + 4) % 4
        elif action == 2:  # Right turn
            new_dir_idx = (current_dir_idx + 1) % 4

        self.direction = DIRECTIONS[new_dir_idx]

        # --- Move snake ---
        head = self.snake[0]
        new_head = Point(head.x + self.direction.x, head.y + self.direction.y)
        self.snake.insert(0, new_head)
        self.head = new_head

        # --- Check game status ---
        # 1. Collision (wall or body)
        if self._is_collision(self.head):
            self.game_over = True
            reward = REWARD_DEATH
            return self._get_state(), reward, self.game_over, self.score

        # 2. Hunger (too many steps without food)
        if self.steps_since_last_food > self.max_steps_without_food:
            self.game_over = True
            reward = REWARD_HUNGER
            return self._get_state(), reward, self.game_over, self.score

        # 3. Food consumption
        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
            self.steps_since_last_food = 0
        else:
            self.snake.pop()

        next_state = self._get_state()
        return next_state, reward, self.game_over, self.score

    def render(self, screen):
        """Draws current game state on Pygame screen."""
        screen.fill(BLACK)
        # Draw snake (blue head, green body)
        for i, segment in enumerate(self.snake):
            color = BLUE if i == 0 else GREEN
            pygame.draw.rect(screen, color, (segment.x * GRID_SIZE, segment.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw food (red)
        if self.food:
            pygame.draw.rect(screen, RED, (self.food.x * GRID_SIZE, self.food.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))