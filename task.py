# task.py - O Jogo
from jax import numpy as jnp
from config import RewardConfig
from enviroment import StateMonad

def reward_and_termination_pipeline(progress, reward_config: RewardConfig, pdata: dict):
    """Calcula a recompensa puramente baseada nos dados extraídos."""
    return (
        StateMonad.pure(pdata)
        .map(
            lambda pdata: {
                **pdata,
                "reward": -reward_config.pos_incentive_gain.update(progress)
                * pdata["position_error"],
            }
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "reward": pdata["reward"]
                        + reward_config.velocity_penalty.update(progress)
                        * jnp.linalg.norm(pdata["joint_vel"], ord=2),
                    },
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "err_tol": reward_config.err_tol.update(progress)},
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "success": pdata["position_error"] < pdata["err_tol"],
            }
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    {
                        **state,
                        "success": jnp.where(pdata["success"], 1.0, 0.0),
                        "step": state["step"] + 1,
                    },
                    pdata,
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,

                    #penalidade proporcional ao numero de passos
                    {**pdata, "reward": pdata["reward"] + state["step"] * reward_config.steps_penalty.update(progress),},
                )
            )
        )
        # Aplicação das Recompensas de Término
        .map(
            lambda pdata: {
                **pdata,
                "done": pdata["success"],

                # Bônus de Sucesso
                "reward": pdata["reward"] + pdata["success"] * reward_config.success_reward.update(progress),
            }
        )
        .map(
            lambda pdata: {
                **pdata,
                "reward": jnp.clip(pdata["reward"], -100.0, 100.0),
            }
        )
    )

def calculate_task_errors(goal_state, pdata: dict):
    """Calcula erros relativos ao alvo atual da tarefa."""
    def fn(state):
        # Lê o alvo que está guardado no estado dinâmico
        current_goal = state["goal"]["goal_position_coordinates"]
        
        error_data = {
            **pdata,
            "position_error": jnp.linalg.norm(current_goal - pdata["tool_position"])
        }
        return state, error_data
    return StateMonad(fn)