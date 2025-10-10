from stable_baselines3 import PPO
import environment

# TO BE CHECKED, IT DOES NOT CONVERGES!!! 

# env params
datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M.csv'
battery_capacity = 100                  # [Wh]
power_idle = 2.5                        # [W]
power_frame = 7.0                       # [W]
delta_time = 15 * 60            # [sec]
proc_interval = 1 * 60      # [sec]
max_irrad = 1200                        # [W/m^2]
pv_efficiency = 0.2
pv_area = 1.0

env = environment.EnergyPVEnv(
        datapath,
        battery_capacity,
        power_idle,
        power_frame,
        delta_time,
        proc_interval,
        max_irrad,
        pv_efficiency,
        pv_area
    )
env.reset(None)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(obs, reward, done)
