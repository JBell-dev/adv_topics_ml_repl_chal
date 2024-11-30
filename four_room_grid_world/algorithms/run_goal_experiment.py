import subprocess

NUMBER_RUNS = 5
is_reward_free = False
is_reward_free = str(is_reward_free)
max_episode_length = 1_000
max_episode_length = str(max_episode_length)

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length]
    subprocess.run(["python", "ppo.py"] + arguments, check=True)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length]
    subprocess.run(["python", "ppo_noisy_net.py"] + arguments, check=True)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length]
    subprocess.run(["python", "ppo_rle_adopted.py"] + arguments, check=True)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--reward-free", is_reward_free, "--max-episode-steps", max_episode_length]
    subprocess.run(["python", "ppo_rnd.py"] + arguments, check=True)
    print(f"Ended run {i}")
