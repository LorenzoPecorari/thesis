from sb3_mas_train import SB3_MAS_Train
from sb3_mas_train_threads import SB3_MAS_Train_Parallelized_Threads
from sb3_mas_train_processes import SB3_MAS_Train_Parallelized_Processes


irradiance_datapaths = [
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    ]
delta_time = 15 * 60
proc_interval = 10 * 60
proc_rate = 20
arrival_rate = 15

num_episodes = 4001

eps_init = 1.0
eps_fin = 0.05
# eps_dec = 0.999
eps_dec = 0.9988

# num_agents = 2
# battery_capacities = [25, 100]
# panel_surfaces = [1.0, 0.5]


# num_agents = 3
# battery_capacities = [25, 100, 50]
# panel_surfaces = [1.0, 0.5, 0.75]

# num_agents = 4
# battery_capacities = [25, 100, 50, 37]
# panel_surfaces = [1.0, 0.5, 0.75, 0.85]

num_agents = 5
battery_capacities = [25, 100, 50, 37, 65]
panel_surfaces = [1.0, 0.5, 0.75, 0.85, 0.65]


power_idle = 2.6
power_max = 6.0

w = 1.0

train_freq = 8

mode = 'cuda'

if __name__ == "__main__":
    trainer1 = SB3_MAS_Train(
        num_agents,
        num_episodes,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        eps_init,
        eps_fin,
        eps_dec,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        train_freq,
        w,
        mode
        )
    
    # trainer.train_with_profiling()

    trainer2 = SB3_MAS_Train_Parallelized_Threads(
        num_agents,
        num_episodes,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        eps_init,
        eps_fin,
        eps_dec,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        train_freq,
        w,
        mode
        )
    
    # trainer.train_with_profiling()

    trainer3 = SB3_MAS_Train_Parallelized_Processes(
        num_agents,
        num_episodes,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        eps_init,
        eps_fin,
        eps_dec,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        train_freq,
        w,
        mode
        )
    
    # # trainer.train_with_profiling()
    trainer3.train()
    trainer2.train()
    trainer1.train()
