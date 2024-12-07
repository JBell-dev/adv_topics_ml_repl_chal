import subprocess

max_episode_length = 1_000
max_episode_length = str(max_episode_length)

# ---

is_reward_free = True
is_reward_free = str(is_reward_free)

arguments = ["--seed", "10", "--goal-distribution", "standard_normal", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length, "--create-demo", "True"]
subprocess.run(["python", "../algorithms/ppo_rle_adopted.py"] + arguments, check=True)

# ---

is_reward_free = False
is_reward_free = str(is_reward_free)

arguments = ["--seed", "0", "--goal-distribution", "standard_normal", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length, "--create-demo", "True"]
subprocess.run(["python", "../algorithms/ppo_rle_adopted.py"] + arguments, check=True)


