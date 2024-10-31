import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class FourRoomGridWorld(gym.Env):
    """
    Adopted from https://www.gymlibrary.dev/content/environment_creation/#subclassing-gym-env.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=50, is_reward_free=False):
        self._is_reward_free = is_reward_free

        assert size % 2 == 0  # Ensure wall can be in the middle
        assert size >= 10  # Ensure minimum size to allow space for walls and holes

        self.size = size + 1  # +1 for the wall
        self.window_size = 1024  # PyGame window size
        self.x_wall_position = size // 2
        self.y_wall_position = size // 2

        # Define wall hole positions
        self.hole_1_position = np.array([self.x_wall_position, size // 2 - 5])
        self.hole_2_position = np.array([self.x_wall_position, size // 2 + 5])
        self.hole_3_position = np.array([size // 2 - 5, self.y_wall_position])
        self.hole_4_position = np.array([size // 2 + 5, self.y_wall_position])

        self.observation_space = spaces.Box(0, size, shape=(2,), dtype=int)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._target_location

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def _position_is_not_in_wall(self, pos):
        # Check if the position is either not on the wall line or exactly in one of the wall holes
        return (
                pos[0] != self.x_wall_position and pos[1] != self.y_wall_position
        ) or (
                np.array_equal(pos, self.hole_1_position)
                or np.array_equal(pos, self.hole_2_position)
                or np.array_equal(pos, self.hole_3_position)
                or np.array_equal(pos, self.hole_4_position)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([self.size - 1, 0], dtype=int)  # Top right corner
        self._start_location = self._agent_location
        self._target_location = np.array([0, self.size - 1], dtype=int)  # Bottom left corner

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Calculate potential new position
        direction = self._action_to_direction[action]
        new_position = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Update the agent’s position only if the new position is valid
        if self._position_is_not_in_wall(new_position):
            self._agent_location = new_position

        # Check if the episode has terminated (i.e., agent reached the target)
        terminated = False
        if not self._is_reward_free:  # No goal in reward-free exploration
            terminated = np.array_equal(self._agent_location, self._target_location)

        reward = 0

        if not self._position_is_not_in_wall(new_position):
            reward = -1

        new_position = self._agent_location + direction
        if new_position[0] < 0 or new_position[0] > self.size - 1 or new_position[1] < 0 or new_position[1] > self.size - 1:
            reward = -1

        if terminated:
            reward = 1

        # Get updated observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Render updated frame if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw wall cells, skipping the hole positions
        for x in range(self.size):
            for y in range(self.size):
                if (
                        (x == self.x_wall_position or y == self.y_wall_position) and
                        not (np.array_equal([x, y], self.hole_1_position) or
                             np.array_equal([x, y], self.hole_2_position) or
                             np.array_equal([x, y], self.hole_3_position) or
                             np.array_equal([x, y], self.hole_4_position))
                ):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),  # Black for walls
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Draw the start cell in red
        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # Red for start
            pygame.Rect(
                pix_square_size * self._start_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the target cell in green
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Green for target
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue for agent
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
