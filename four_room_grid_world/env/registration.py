from four_room_grid_world.env.FourRoomGridWorld import FourRoomGridWorld  # Do not remote this import

from gym.envs.registration import register

register(
    id='advtop/FourRoomGridWorld-v0',
    entry_point='four_room_grid_world.env.FourRoomGridWorld:FourRoomGridWorld',
    max_episode_steps=300,
)