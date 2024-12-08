import pandas as pd
import matplotlib.pyplot as plt

df_rle = pd.read_csv('extrinsic_returns_rle_running_avg.csv')
df_ppo = pd.read_csv('extrinsic_returns_ppo_running_avg1.csv')
df_noisy = pd.read_csv('extrinsic_returns_ppo_rndr.csv')


plt.figure(figsize=(15, 8))
plt.plot(df_rle['Episode'], df_rle['Extrinsic Return'], 
         label='RLE', linewidth=1.5, color='#2E86C1')
plt.plot(df_ppo['Episode'], df_ppo['Extrinsic Return'], 
         label='PPO', linewidth=1.5, color='#E74C3C')
plt.plot(df_noisy['Episode'], df_noisy['Extrinsic Return'],     
         label='PPO + WHITE NOISE', linewidth=1.5, color='#F1C40F')

plt.title('RLE vs PPO vs Noisy: Extrinsic Returns Over Episodes (Running Average)', 
          fontsize=14, pad=15)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Extrinsic Return', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()

plt.savefig('comparison_plot.pdf', dpi=300, bbox_inches='tight')

