
import jax
import jax.numpy as jnp
from typing import cast
from networks import NetworksSettings, NetworkParameters
from enviroment import StateMonad
from robot import RobotSharedData
from utils import cont_sample_beta

def get_action(
    network_settings: NetworksSettings, network_parameters: NetworkParameters
):
    def fn(state):
        last_obs = state["obs"]
        rng1, rng2 = jax.random.split(state["rng"])

        alpha, beta = network_settings.actor.apply(network_parameters.actor, last_obs)
        alpha = cast(jax.Array, alpha)
        beta = cast(jax.Array, beta)

        action, logprob = cont_sample_beta(rng1, alpha, beta)

        new_state = {**state, "rng": rng2}
        return new_state, {"action": action, "logprob": logprob}

    return StateMonad(fn)


def get_ctrl(rsd: RobotSharedData, pdata,  alpha=0.1, max_step_rads = 0.05 ):
   
    # Ação vira um delta de movimento seguro. Ex: max 0.05 radianos por frame
    def fn(state):
        # smoothed no intervalo [0, 1]
        smoothed_action = alpha * state["last_action"] + (1 - alpha) * pdata["action"]
        delta = smoothed_action * max_step_rads


        ctrl = state["mjx_data"].qpos + delta

        new_state = {**state, "last_action": smoothed_action}
        return new_state, {**pdata, "ctrl": jnp.clip(ctrl, rsd.lowers, rsd.uppers)}

    return StateMonad(fn)
    
def shape_return(pdata):
    def fn(state):
        state = {
            **state,
            "ctrl_norm": jnp.linalg.norm(pdata["ctrl"], ord=2),
        }

        # protege a rede neural se a física quebrar
        is_healthy = jnp.isfinite(pdata["reward"]) & jnp.isfinite(pdata["logprob"])
        done = pdata["done"] | (~is_healthy)
        safe_reward = jnp.where(is_healthy, pdata["reward"], -50.0)

        data = {
            "obs": pdata["obs"],
            "action": pdata["action"],
            "reward": safe_reward,
            "logprob": jnp.nan_to_num(pdata["logprob"], nan=0.0),
            "done": done,
        }
        return state, data

    return StateMonad(fn)