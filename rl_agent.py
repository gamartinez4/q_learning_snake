#rl_agent.py

import numpy as np
import random
import json
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.9995, min_exploration_rate=0.01,
                 q_table_file='q_table_snake.json'):
        """
        Initializes the Q-Learning agent.

        Args:
            state_size (int): Dimensionality of state space (length of state tuple).
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate (alpha).
            discount_factor (float): Discount factor (gamma).
            exploration_rate (float): Initial exploration rate (epsilon).
            exploration_decay (float): Epsilon decay factor per episode.
            min_exploration_rate (float): Minimum exploration rate.
            q_table_file (str): Filename for saving/loading Q-Table.
        """
        if not isinstance(state_size, int) or state_size <= 0:
             raise ValueError("state_size must be a positive integer.")
        if not isinstance(action_size, int) or action_size <= 0:
             raise ValueError("action_size must be a positive integer.")

        self.state_size = state_size
        self.action_size = action_size

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        self.q_table_file = q_table_file
        # Q-Table loading happens here, when initializing the agent
        self.q_table = self._load_q_table()
        print(f"Agent initialized. Q-Table size: {len(self.q_table)} states.")


    def _load_q_table(self):
        """Loads Q-Table from JSON file if exists, otherwise creates an empty one."""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'r') as f:
                    q_table_str_keys = json.load(f)
                    # Convert string keys (JSON) to tuples (Python) and values to numpy arrays
                    q_table_loaded = {eval(k): np.array(v, dtype=np.float32) for k, v in q_table_str_keys.items()}
                    print(f"Q-Table loaded from {self.q_table_file}")

                    # Basic dimension validation (optional but recommended)
                    for state_tuple, q_values in q_table_loaded.items():
                         if not isinstance(state_tuple, tuple) or len(state_tuple) != self.state_size:
                              print(f"Warning: State '{state_tuple}' in Q-table has length {len(state_tuple)}, expected {self.state_size}. Resetting Q-table.")
                              return {} # Reset if state size doesn't match
                         if len(q_values) != self.action_size:
                             print(f"Warning: State '{state_tuple}' in Q-table has {len(q_values)} actions, expected {self.action_size}. Resetting Q-table.")
                             return {} # Reset if action count doesn't match
                    return q_table_loaded
            except (json.JSONDecodeError, TypeError, ValueError, SyntaxError) as e:
                print(f"Error loading Q-Table from {self.q_table_file} or invalid format: {e}. Creating new one.")
                return {} # Empty dict if error occurs
            except Exception as e:
                 print(f"Unexpected error loading Q-Table: {e}. Creating new one.")
                 return {}
        else:
            print(f"Q-Table file '{self.q_table_file}' not found. Creating new one.")
            return {} # Empty dict if file doesn't exist

    def save_q_table(self):
        """Saves current Q-Table to JSON file."""
        try:
            # Pre-validation before saving
            valid_q_table = {}
            for k, v in self.q_table.items():
                if not isinstance(k, tuple) or len(k) != self.state_size:
                    print(f"Error: Invalid state key found: {k}. Will be omitted when saving.")
                    continue
                if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.action_size:
                    print(f"Error: Invalid Q-value for state {k}: {v}. Will be omitted when saving.")
                    continue
                # Ensure Q-values are finite (replace NaN/inf with 0)
                if not np.all(np.isfinite(v)):
                    print(f"Warning: Non-finite Q-values detected for state {k}. Replacing with zeros.")
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                valid_q_table[k] = v

            # Convert tuple keys to strings for JSON and ndarray values to lists
            q_table_str_keys = {str(k): v.tolist() for k, v in valid_q_table.items()}

            with open(self.q_table_file, 'w') as f:
                json.dump(q_table_str_keys, f, indent=4) # indent=4 for better readability
            print(f"Q-Table saved to {self.q_table_file} ({len(valid_q_table)} states).")
        except IOError as e:
            print(f"I/O error saving Q-Table to {self.q_table_file}: {e}")
        except Exception as e:
            print(f"Unexpected error saving Q-Table: {e}")

    def _get_q_values(self, state):
        """Gets Q-values for given state. Creates an entry if doesn't exist."""
        # Ensure state is a tuple (required dictionary key)
        if not isinstance(state, tuple):
             # Try to convert lists or numpy arrays to tuple
             try:
                 state = tuple(state)
             except TypeError:
                 raise TypeError(f"State must be convertible to tuple, but received type {type(state)}")

        # If state not in table, initialize with zeros
        if state not in self.q_table:
            # print(f"New state found: {state}. Initializing Q-values.") # Debug
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)

        # Return a copy to prevent accidental modifications outside learn()
        return self.q_table[state].copy()

    def choose_action(self, state, training=True):
        """
        Chooses an action using epsilon-greedy policy.
        - If training is True and random number < epsilon, chooses random action (exploration).
        - Otherwise, chooses action with highest Q-value for current state (exploitation).
        """
        # Ensure state is a tuple
        if not isinstance(state, tuple): state = tuple(state)

        if training and random.uniform(0, 1) < self.epsilon:
            # Exploration
            action = random.randint(0, self.action_size - 1)
            # print(f"Epsilon={self.epsilon:.3f} -> Exploring: Action {action}") # Debug
        else:
            # Exploitation
            q_values = self._get_q_values(state) # Gets or initializes Q-values for state
            action = np.argmax(q_values)
            # print(f"Epsilon={self.epsilon:.3f} -> Exploiting: Q={q_values}, Action {action}") # Debug
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Updates Q-value for (state, action) pair using Q-learning rule.
        Also handles epsilon decay if episode has ended.
        """
        # Ensure states are tuples
        if not isinstance(state, tuple): state = tuple(state)
        if not isinstance(next_state, tuple): next_state = tuple(next_state)

        # Get current and next state Q-values
        # Using _get_q_values to ensure state exists in table
        q_values_state = self._get_q_values(state) # Returns copy if exists, or zeros array
        q_values_next_state = self._get_q_values(next_state)

        # Current Q-value for taken action
        current_q = q_values_state[action]

        # Calculate target Q-value
        if done:
            target_q = reward # If terminal state, future value is 0
        else:
            # Q-Learning: target = reward + gamma * max_a'(Q(s', a'))
            max_next_q = np.max(q_values_next_state)
            target_q = reward + self.gamma * max_next_q

        # Calculate new Q-value using update formula
        new_q = current_q + self.lr * (target_q - current_q)

        # Check if new_q is valid before assigning
        if not np.isfinite(new_q):
             print(f"Warning: Calculated non-finite Q-value ({new_q}). Won't update for state {state}, action {action}.")
        else:
             # Update Q-value in original table (not the copy)
             # Ensure state exists before assigning (though _get_q_values already did this)
             if state not in self.q_table:
                 self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
             self.q_table[state][action] = new_q

        # Epsilon decay AT END of episode (when done=True)
        if done:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                # Ensure epsilon doesn't fall below minimum
                self.epsilon = max(self.min_epsilon, self.epsilon)
                # print(f"Episode ended. Epsilon decay: {self.epsilon:.4f}") # Debug