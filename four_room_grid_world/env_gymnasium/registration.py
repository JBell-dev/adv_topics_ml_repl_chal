from four_room_grid_world.env.FourRoomGridWorld import FourRoomGridWorld  # Do not remote this import

from gymnasium.envs.registration import register

register(
    id='advtop/FourRoomGridWorld-v0',
    entry_point='four_room_grid_world.env_gymnasium.FourRoomGridWorld:FourRoomGridWorld',
    max_episode_steps=1000,
)