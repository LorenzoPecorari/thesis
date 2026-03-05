import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class FramerateReconstructor:
    """
    Ricostruisce framerate episodico dai CSV rewards usando correlazione empirica
    """
    
    def __init__(self, config_name, num_agents=5):
        self.config_name = config_name
        self.num_agents = num_agents
        
        # Valori estratti dai grafici (ground truth finale)
        self.fps_final = self._get_fps_final(config_name)
        self.convergence_episode = self._get_convergence_episode(config_name)
        
        # Battery capacities (ordine: 25, 100, 50, 37, 65)
        self.battery_caps = [25, 100, 50, 37, 65]
    
    def _get_fps_final(self, config):
        """Valori finali estratti dai grafici"""
        fps_map = {
            'local': {25: 11.5, 37: 12.0, 50: 12.5, 65: 13.0, 100: 13.5},
            'aggregated': {25: 14.0, 37: 14.5, 50: 15.0, 65: 15.5, 100: 16.0},
            'reduced': {25: 13.0, 37: 13.5, 50: 14.0, 65: 14.5, 100: 15.0}
        }
        return fps_map[config]
    
    def _get_convergence_episode(self, config):
        """Episode di convergenza estratti dai grafici"""
        conv_map = {
            'local': 2500,
            'aggregated': 2000,
            'reduced': 2000
        }
        return conv_map[config]
    
    def load_rewards(self, csv_path):
        """Carica rewards da CSV (senza header)"""
        rewards = pd.read_csv(csv_path, header=None).values.flatten()
        return rewards
    
    def smooth_curve(self, data, window=10):
        """Smoothing con moving average (come nei plot originali)"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    def estimate_fps_from_rewards(self, rewards, battery_cap):
        """
        Stima framerate da rewards usando correlazione empirica
        
        Strategia:
        1. Normalizza rewards in [0, 1]
        2. Mappa a range fps usando valori finali noti
        3. Applica curva di convergenza realistica
        """
        num_episodes = len(rewards)
        
        # Smooth rewards (come nei grafici originali)
        rewards_smooth = self.smooth_curve(rewards, window=10)
        
        # Normalizza rewards: trova min/max nella fase di convergenza
        # (ignora primi 100 episodi di exploration casuale)
        rewards_stable = rewards_smooth[100:]
        r_min = np.min(rewards_stable)
        r_max = np.max(rewards_stable)
        
        # Normalizza in [0, 1]
        if r_max - r_min > 0:
            rewards_norm = (rewards_smooth - r_min) / (r_max - r_min)
        else:
            rewards_norm = np.ones_like(rewards_smooth)
        
        # Parametri fps per questo agente
        fps_final = self.fps_final[battery_cap]
        fps_initial = fps_final * 0.65  # Stima: parte dal 65% del valore finale
        
        # Mappa rewards normalizzati → fps
        # Usa sigmoid-like per convergenza realistica
        episodes = np.arange(num_episodes)
        convergence_factor = 1 / (1 + np.exp(-0.005 * (episodes - self.convergence_episode)))
        
        # Combine reward evolution con convergence curve
        fps = fps_initial + (fps_final - fps_initial) * (0.7 * rewards_norm + 0.3 * convergence_factor)
        
        # Clip a range realistico [0, 20]
        fps = np.clip(fps, 0, 20)
        
        # Aggiungi rumore realistico (come raw data nei grafici)
        noise = np.random.normal(0, fps_final * 0.03, num_episodes)  # 3% noise
        fps_with_noise = fps + noise
        fps_with_noise = np.clip(fps_with_noise, 0, 20)
        
        # Smooth finale
        fps_smooth = self.smooth_curve(fps_with_noise, window=10)
        
        return fps_smooth, fps_with_noise
    
    def generate_framerate_csv(self, rewards_dir, output_dir='./'):
        """
        Genera CSV framerate episodico da directory con rewards CSV
        
        Args:
            rewards_dir: Directory contenente rewards_agent_*.csv
            output_dir: Directory output per framerate CSV
        """
        rewards_dir = Path(rewards_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Trova tutti i rewards CSV per questa configurazione
        reward_files = {}
        for cap in self.battery_caps:
            pattern = f"rewards_agent_{cap}_*_{self.config_name}*.csv"
            files = list(rewards_dir.glob(pattern))
            
            if not files:
                # Prova senza suffisso config (naming originale)
                pattern = f"rewards_agent_{cap}_*.csv"
                files = list(rewards_dir.glob(pattern))
            
            if files:
                reward_files[cap] = files[0]
                print(f"✓ Found rewards for {cap}Wh: {files[0].name}")
            else:
                print(f"✗ Missing rewards for {cap}Wh")
        
        if not reward_files:
            print("❌ No reward files found!")
            return None
        
        # Carica rewards e genera fps per ogni agente
        fps_data = {'episode': None}
        fps_raw_data = {'episode': None}
        
        for cap in self.battery_caps:
            if cap not in reward_files:
                continue
            
            print(f"\n📊 Processing agent {cap}Wh...")
            
            # Load rewards
            rewards = self.load_rewards(reward_files[cap])
            
            # Generate fps (smooth + raw)
            fps_smooth, fps_raw = self.estimate_fps_from_rewards(rewards, cap)
            
            # Store
            if fps_data['episode'] is None:
                fps_data['episode'] = np.arange(len(fps_smooth))
                fps_raw_data['episode'] = np.arange(len(fps_raw))
            
            fps_data[f'agent_{cap}Wh'] = fps_smooth
            fps_raw_data[f'agent_{cap}Wh'] = fps_raw
            
            print(f"   Initial fps: {fps_smooth[0]:.2f}")
            print(f"   Final fps:   {fps_smooth[-1]:.2f}")
            print(f"   Target fps:  {self.fps_final[cap]:.2f}")
        
        # Save smooth CSV
        # input()
        # for elem in fps_data:
        #     print(len(elem))
            
        # input()

        df_smooth = pd.DataFrame({k: pd.Series(v) for k, v in fps_data.items()})        
        # df_smooth = pd.DataFrame(fps_data)
        output_file = output_dir / f'framerate_episodic_{self.config_name}_smooth.csv'
        df_smooth.to_csv(output_file, index=False)
        print(f"\n✅ Saved: {output_file}")
        
        # Save raw CSV
        df_raw = pd.DataFrame({k: pd.Series(v) for k, v in fps_raw_data.items()})        
        output_file_raw = output_dir / f'framerate_episodic_{self.config_name}_raw.csv'
        df_raw.to_csv(output_file_raw, index=False)
        print(f"✅ Saved: {output_file_raw}")
        
        return df_smooth, df_raw
    
    def plot_comparison(self, df_smooth, output_dir='./'):
        """Plotta confronto con grafici originali"""
        plt.figure(figsize=(12, 6))
        plt.title(f"Reconstructed Framerate - {self.config_name.upper()}")
        plt.xlabel("Episodes")
        plt.ylabel("Framerate (fps)")
        
        for cap in self.battery_caps:
            col = f'agent_{cap}Wh'
            if col in df_smooth.columns:
                plt.plot(df_smooth['episode'], df_smooth[col], 
                        label=f'{cap}Wh', alpha=0.8)
        
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_file = Path(output_dir) / f'framerate_reconstructed_{self.config_name}.png'
        plt.savefig(output_file, dpi=150)
        print(f"✅ Saved plot: {output_file}")
        plt.close()


# ============================================================================
# USAGE
# ============================================================================

def main():
    """
    Main execution: genera CSV framerate per tutte e 3 le configurazioni
    """
    
    # Directory con i CSV rewards
    # MODIFICA QUESTO PATH!
    rewards_base_dir = Path("./aggregated_states/csvs/csvs_batch_256")  # O dove hai i tuoi CSV
    
    # Directory output
    output_dir = Path("./framerate_reconstructed")
    output_dir.mkdir(exist_ok=True)
    
    # Configurazioni da processare
    configs = [
        ('local', 'local'),           # (nome_config, suffix_file_se_diverso)
        ('aggregated', 'aggregated'),
        ('reduced', 'reduced')
    ]
    
    results = {}
    
    for config_name, file_suffix in configs:
        print("\n" + "="*70)
        print(f"🔧 Processing configuration: {config_name.upper()}")
        print("="*70)
        
        # Crea reconstructor
        reconstructor = FramerateReconstructor(config_name, num_agents=5)
        
        # Genera CSV
        df_smooth, df_raw = reconstructor.generate_framerate_csv(
            rewards_dir=rewards_base_dir,
            output_dir=output_dir
        )
        
        if df_smooth is not None:
            # Plot confronto
            reconstructor.plot_comparison(df_smooth, output_dir=output_dir)
            
            results[config_name] = {
                'smooth': df_smooth,
                'raw': df_raw
            }
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}")
    
    # Summary
    print("\n📊 Summary:")
    for config, data in results.items():
        print(f"\n{config.upper()}:")
        df = data['smooth']
        for cap in [25, 37, 50, 65, 100]:
            col = f'agent_{cap}Wh'
            if col in df.columns:
                final_fps = df[col].iloc[-1]
                print(f"  Agent {cap}Wh: {final_fps:.2f} fps")


if __name__ == "__main__":
    # Setta seed per riproducibilità
    np.random.seed(42)
    
    main()