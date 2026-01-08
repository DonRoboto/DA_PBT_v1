#!/usr/bin/env python3
"""
TITULO: Entrenamiento PPO con Population Based Training (PBT) en BipedalWalker-v3
AUTOR: y0r5
FECHA: 2026-01-08

DESCRIPCIÓN:
Este script ejecuta un experimento de Aprendizaje por Refuerzo utilizando PPO.
Se emplea PBT para optimizar dinámicamente los hiperparámetros.


HIPERPARÁMETROS MUTABLES:
- Learning Rate (lr)
- Train Batch Size
- Lambda (GAE)
- Clip Parameter

SALIDA:
- Checkpoints de los mejores modelos.
- CSV consolidado de métricas.
"""

import os
import random
import shutil
import time
import pandas as pd
import ray
from datetime import datetime
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining

# --- CONFIGURACIÓN GLOBAL ---
EXPERIMENTS = 1
ENV_NAME = "BipedalWalker-v3"
ALGO = "PPO"
NUM_SAMPLES = 8           # Tamaño de la población
MAX_TIMESTEPS = 1_000_000 # Duración del entrenamiento
METRIC_TARGET = "env_runners/episode_reward_mean" 

# --- FUNCIONES AUXILIARES ---

def explore(config):
    """
    Función de exploración/mutación para PBT.
    """
    # 1. Asegurar consistencia: Batch > Minibatch * 2
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    
    # 2. Asegurar divisibilidad para evitar warnings de 'dropped samples'
    remainder = config["train_batch_size"] % config["sgd_minibatch_size"]
    if remainder != 0:
        config["train_batch_size"] = config["train_batch_size"] - remainder

    # 3. Límites lógicos para Lambda
    if config["lambda"] > 1.0: 
        config["lambda"] = 1.0
    if config["lambda"] < 0.0:
        config["lambda"] = 0.0
        
    # 4. Conversión estricta a enteros
    config["train_batch_size"] = int(config["train_batch_size"])
    config["sgd_minibatch_size"] = int(config["sgd_minibatch_size"])
    
    # NOTA: num_sgd_iter ya no se procesa aquí porque es fijo.
    
    return config

# --- CONFIGURACIÓN DEL SCHEDULER PBT ---
pbt = PopulationBasedTraining(
    time_attr="timesteps_total",
    metric=METRIC_TARGET, 
    mode="max",
    perturbation_interval=50_000,
    resample_probability=0.25,
    quantile_fraction=0.25,
    
    # Hiperparámetros sujetos a mutación
    # NOTA: 'num_sgd_iter' ELIMINADO de aquí.
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 0.99),
        "clip_param": lambda: random.uniform(0.1, 0.4),
        "lr": lambda: random.uniform(1e-5, 1e-3),
        "train_batch_size": lambda: random.randint(2000, 40000), 
    },
    custom_explore_fn=explore,
)

               
                
# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    try:
        for k in range(EXPERIMENTS):
            seed = random.randint(1111, 9999)
            inicio = time.time()
            
            timelog = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{timelog}_{ALGO}_pbt_{ENV_NAME}_seed{seed}"
            output_dir = os.path.join("data", experiment_name)
            
            print(f"\n>>> INICIANDO EXPERIMENTO: {experiment_name}")
            print(f">>> Seed: {seed} | Población: {NUM_SAMPLES} Agentes")

            analysis = run(
                ALGO,
                name=experiment_name,
                scheduler=pbt,
                verbose=1,
                num_samples=NUM_SAMPLES,
                reuse_actors=False,
                checkpoint_config={
                    "checkpoint_frequency": 20, 
                    "checkpoint_at_end": True,
                    "num_to_keep": 5, 
                },
                stop={"timesteps_total": MAX_TIMESTEPS},
                config={
                    # Recursos
                    "num_workers": 1,
                    "num_envs_per_worker": 10, # Vectorización agresiva para compensar 1 solo worker
                    "num_gpus": 0.125,         # 1 GPU dividida entre 8 procesos
                    
                    # Entorno
                    "env": ENV_NAME,
                    "log_level": "WARN",
                    "seed": seed,            
                    "observation_filter": "MeanStdFilter", 
                    
                    # Modelo
                    "model": {
                        "fcnet_hiddens": [256, 256], 
                        "free_log_std": True,
                    },
                    
                    # --- HIPERPARÁMETROS ---
                    
                    # FIJO: Valor estándar recomendado para PPO
                    "num_sgd_iter": 30, 
                    
                    "kl_coeff": 1.0, 
                    "sgd_minibatch_size": 256,
                    
                    # Espacio de búsqueda inicial (mutables)
                    "lambda": sample_from(lambda spec: random.uniform(0.9, 0.99)),
                    "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.4)),
                    "lr": sample_from(lambda spec: random.uniform(1e-5, 1e-3)),
                    "train_batch_size": sample_from(lambda spec: random.randint(2000, 40000)),
                    
                    # Optimizaciones
                    "compress_observations": True,
                },
            )
            

            fin = time.time()
            print(f"\n>>> Entrenamiento finalizado en {(fin - inicio)/60:.2f} minutos.")
            
            # --- PROCESAMIENTO DE RESULTADOS ---
            print(">>> Procesando DataFrames y Checkpoints...")
            results = pd.DataFrame()
            
            cols_to_keep = [
                "trial_id", "config/seed", "timesteps_total", "time_total_s",
                METRIC_TARGET, "env_runners/episode_len_mean",
                "config/lambda", "config/lr", "config/train_batch_size", 
                "config/clip_param"                 
            ]

            for trial_id, df in analysis.trial_dataframes.items():
                valid_cols = [c for c in cols_to_keep if c in df.columns]
                subset_df = df[valid_cols].copy()
                subset_df["trial_id"] = trial_id
                results = pd.concat([results, subset_df], ignore_index=True)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            results.to_csv(os.path.join(output_dir, f"results_seed{seed}.csv"), index=False)
            
            # --- EXTRACCIÓN DE CHECKPOINTS ---
            for i, trial in enumerate(analysis.trials):
                try:
                    checkpoint = analysis.get_best_checkpoint(trial, metric=METRIC_TARGET, mode="max")
                    
                    if checkpoint:
                        folder_name = f"Trial_{i}_{trial.trial_id[:6]}"
                        dest_dir = os.path.join(output_dir, folder_name)
                        shutil.copytree(checkpoint.path, dest_dir, dirs_exist_ok=True)
                        print(f"    -> [OK] Trial {trial.trial_id[:6]} guardado.")
                    else:
                        print(f"    -> [SKIP] Trial {trial.trial_id[:6]} sin checkpoint válido.")
                        
                except Exception as e:
                    print(f"    -> [ERROR] Fallo al guardar Trial {trial.trial_id}: {e}")

    except Exception as e:
        print(f"\n!!! ERROR CRÍTICO DURANTE LA EJECUCIÓN: {e}")
        
    finally:
        print("\n>>> Cerrando Ray...")
        ray.shutdown()
