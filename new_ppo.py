from os import wait
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import utils as mu
from functools import partial
from typing import Dict, Any, cast
from utils import  beta_entropy
from robot import get_goal, obs_pipeline
from mujoco import mjx
from dataclassutils import RunningParameters, TrainingSettings, BatchedBuffer, NetworkParameters
from enviroment import StateMonad


def rollout_step(
    progress,
    step_fn,
    runpar: RunningParameters,
    state: Dict[str, Any],
    obs_buffer: jax.Array,
    action_buffer: jax.Array,
    reward_buffer: jax.Array,
    logprob_buffer: jax.Array,
    ptr: jax.Array,
    done_flag: jax.Array,
    stop_flag: jax.Array,
):
    """
    Dá um step de rollout para um único ambiente

    Parameters
    ----------

    pipeline: StateMonad
        Pipeline de ajuste de objetivo, tomada de ação pelo agente, coleta de recompensas e formação do espaço de observação.
    """

    # Caso stop_flag esteja como False
    def do_step(carry):
        _state, _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr, _done_flag, _stop_flag = carry

        # executa o ambiente
        _state, data = step_fn(progress, _state, runpar)


        # adiciona o dado no buffer
        _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr = BatchedBuffer.push(
            _obs_buffer,
            _action_buffer,
            _reward_buffer,
            _logprob_buffer,
            data["obs"],
            data["action"],
            data["reward"],
            data["logprob"],
            _ptr
        )

        _done_flag = data["done"] > 0.5 
        _stop_flag = _done_flag | (_state["step"] >= _reward_buffer.shape[0])

        return _state, _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr, _done_flag, _stop_flag

    # Caso stop_flag esteja como True
    def no_step(carry):  #
        return carry

    #return do_step((state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag))

    return jax.lax.cond(
        stop_flag, no_step, do_step, (state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag)
    )

def rollout(
    settings: TrainingSettings,
    network_params: NetworkParameters,
    init_state: Dict[str, Any],
    buffer: BatchedBuffer,
    runpar: RunningParameters,
):
    # Match the structure of your 'state' dictionary exactly
    state_in_axes = {
        'last_action': 0,
        'goal': {
            'goal_position_coordinates': 0,
            'goal_position_velocities': 0,
            'goal_orientation_coordinates': 0, # or 0 if batched
        },
        'mjx_data': 0, 
        'obs': 0,
        'rng': 0,
        'step': 0,
        'success_count':0,
        "err":0
    }

    step_fn = settings.step_fn_creator(settings, network_params)

    vmap_rollout_step = jax.vmap(
        partial(rollout_step, runpar.progress.value, step_fn, runpar),
        in_axes=(
            state_in_axes,  # Arg 0: state (was Arg 1 in your version)
            0,               # Arg 1: obs_buffer
            0,               # Arg 2: action_buffer
            0,               # Arg 3: reward_buffer
            0,               # Arg 4: logprob_buffer
            0,               # Arg 5: ptr
            0,               # Arg 6: done_flag
            0,               # Arg 7: stop_flag
        ),
    )

    def scan_fn(carry, _):
        state, buffer = carry
        state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag = vmap_rollout_step(
            state,
            buffer.obs_buffer,
            buffer.action_buffer,
            buffer.reward_buffer,
            buffer.logprob_buffer,
            buffer.ptr,
            buffer.done_flag,
            buffer.stop_flag
        )

        buffer = BatchedBuffer(obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag)
        return (state, buffer), None

    (final_state, final_buffer), _ = jax.lax.scan(
        scan_fn, (init_state, buffer), None, length=settings.rollout_steps
    )
    return final_state, final_buffer


def general_advantage_estimator(
    settings: TrainingSettings,
    network_parameters: NetworkParameters,
    obs_buffer: jax.Array,    # Shape: (N + 1, *obs) 
    reward_buffer: jax.Array, # Shape: (N + 1,)
    ptr: jax.Array,           # Valor entre 0 e N
    done_flag: jax.Array,     # Flag de terminação no passo ptr-1
):
    gamma = settings.gamma
    lam = settings.gae_lambda

    N = reward_buffer.shape[0] -1

    values = settings.network_settings.critic.apply(
        network_parameters.critic, obs_buffer
    )
    values = jnp.squeeze(cast(jax.Array, values))

    # Se ptr=N, values[ptr] acessa o índice N (o último do buffer N+1)
    bootstrap_value = jnp.where(done_flag, 0.0, values[ptr])

    def gae_scan_fn(carry, t):
        gae_next, next_val_from_carry = carry
        
        # Injeta o bootstrap exatamente na borda da pilha de dados
        is_last_step = t == (ptr - 1)
        actual_next_val = cast(jax.Array, jnp.where(is_last_step, bootstrap_value, next_val_from_carry))
        
        # O done_buffer[t_idx] aqui só seria necessário se você tivesse 
        # múltiplos episódios dentro do mesmo buffer. No seu caso de "stack",
        # apenas a done_flag no final importa para o bootstrap.
        
        delta = reward_buffer[t] + gamma * actual_next_val - values[t]
        gae = delta + gamma * lam * gae_next
        
        return (gae, values[t]), gae

    #Scan Reverso sobre os N passos de transição
    initial_carry = (0.0, 0.0)
    _, advantages = jax.lax.scan(
        gae_scan_fn,
        initial_carry,
        jnp.arange(N),
        reverse=True
    )

    valid_mask = (jnp.arange(N) < ptr).astype(jnp.float32)
    advantages = advantages * valid_mask
    returns = (advantages + values[:-1]) * valid_mask

    return advantages, returns


def ppo_loss(
    params: NetworkParameters,
    settings: TrainingSettings,
    batch_obs,          # shape: (num_envs, max_steps +1, *obs_shape)
    batch_actions,      # shape: (num_envs, max_steps +1, *action_shape)
    batch_advantages,   # shape: (num_envs, max_steps)
    batch_returns,      # shape: (num_envs, max_steps)
    old_log_probs,      # shape: (num_envs, max_steps +1)
    batch_ptr,          # ADICIONADO: shape (num_envs,) vindo do buffer.ptr
    clip_eps=0.2,
    c1=0.8,
    c2=0.01,
    min_alpha_beta=1.0,
):
    batch_advantages = jax.lax.stop_gradient(batch_advantages)
    batch_returns = jax.lax.stop_gradient(batch_returns)

    batch_obs = jax.lax.stop_gradient(batch_obs[:, :-1, :])
    batch_actions = jax.lax.stop_gradient(batch_actions[:, :-1, :])
    old_log_probs = jax.lax.stop_gradient(old_log_probs[:, :-1])

    # --- NOVO: CRIAR MÁSCARA DE PASSOS VÁLIDOS ---
    max_steps = batch_advantages.shape[1]
    steps_arr = jnp.arange(max_steps)

    # Compara a matriz de passos com o ponteiro de cada ambiente
    valid_mask = steps_arr[None, :] < batch_ptr[:, None]  # shape: (num_envs, max_steps)
    total_valid_elements = jnp.sum(valid_mask) + 1e-6     # Evita divisão por zero
    # ----------------------------------------------

    # forward 
    networks = settings.network_settings
    logits = cast(jax.Array, networks.actor.apply(params.actor, batch_obs))
    values = cast(jax.Array, networks.critic.apply(params.critic, batch_obs))

    # parametrização
    alpha_logits, beta_logits = jnp.split(logits, 2, axis=-1)
    alpha = jnp.clip(jax.nn.softplus(alpha_logits) + min_alpha_beta, 1.0, 100.0)
    beta  = jnp.clip(jax.nn.softplus(beta_logits) + min_alpha_beta, 1.0, 100.0)

    # logprobs
    clipped_actions = jnp.clip(batch_actions, 1e-6, 1 - 1e-6)
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    logprobs = jnp.sum(logprobs, axis=2)

    # ratio
    ratio = jnp.exp(logprobs - old_log_probs)

    # PPO objective (Modificado para aplicar a máscara)
    unclipped = ratio * batch_advantages
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
    raw_policy_loss = -jnp.minimum(unclipped, clipped)
    policy_loss = jnp.sum(raw_policy_loss * valid_mask) / total_valid_elements

    # Value loss (Modificado para ignorar passos inválidos)
    raw_value_loss = (batch_returns - values) ** 2
    value_loss = c1 * (jnp.sum(raw_value_loss * valid_mask) / total_valid_elements)

    # Entropy (Modificado para ignorar passos inválidos)
    raw_entropy = beta_entropy(alpha, beta).sum(axis=2)
    entropy = c2 * (jnp.sum(raw_entropy * valid_mask) / total_valid_elements)

    total_loss = policy_loss + value_loss - entropy
    return total_loss, {"entropy": entropy, "policy_loss": policy_loss, "value_loss": value_loss}


##########################################################################


##########################################################################
def train_epochs(
        settings: TrainingSettings,
        network_parameters: NetworkParameters,
        optimizer_state: optax.OptState,
        buffer: BatchedBuffer,
        advantages: jax.Array,
        returns:jax.Array
    ):
        def grad_norm(grads):
            leaves = jax.tree_util.tree_leaves(grads)

            # norma euclidiana (L2) do gradiente
            return jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in leaves]))

        def single_epoch(carry,_):
            _parameters, _optimizer_state = carry
    
            def loss_fn(par):
                return ppo_loss(
                    par,
                    settings,
                    buffer.obs_buffer,
                    buffer.action_buffer,
                    advantages,
                    returns,
                    buffer.logprob_buffer,
                    batch_ptr=buffer.ptr,
                )

            # calcula os gradientes e atualiza os parametros
            (loss_val, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(_parameters)
            updates, new_optim_state = settings.optimizer.update(grads, _optimizer_state)
            new_parameters = cast(NetworkParameters, optax.apply_updates(_parameters, updates))

            new_carry = (new_parameters, new_optim_state)
        
            return new_carry, {
                "loss": loss_val,
                "grad_norm": grad_norm(grads),
                **aux_metrics,
            }

        # após  o scan, teremos o seguinte:
        # metric[key].shape  = (epochs, *metric_shape)
        # Ex: loss.shape  = (epochs, )
        (new_parameters, new_optim_state), metrics = jax.lax.scan(
            single_epoch,
            (network_parameters, optimizer_state),
            jnp.arange(settings.epochs),
        )

        return new_parameters, new_optim_state, metrics

def update_goal(state, progress, settings:TrainingSettings):
    vmapped_get_goal = jax.vmap(partial(get_goal, settings.robot_shared_data.range_config, progress))
    batched_rng, batched_goal = vmapped_get_goal(state["rng"])
    return {**state, "rng": batched_rng, "goal":batched_goal}

jax.jit
def ppo_train(rng: jax.Array, starting_network_params: NetworkParameters, settings: TrainingSettings):
    """The complete, JIT-compiled training function."""

    def get_success_rate(state):
        # Transforma qualquer contagem > 0 em 1 (sucesso) ou 0 (falha)
        # A média dará um valor entre  0.0 e 1.0 (0% a 100%)
        # success.shape: (num_envs, )
        success_count = state["success_count"]
        nsteps = state["step"]

        rate  = jnp.mean(success_count/(nsteps + 1e-6))
        return jax.lax.stop_gradient(rate)

    def collect_rollouts(state, buffer, runpar: RunningParameters, network_params: NetworkParameters):

        # Faz um rollout (usando a função vetorizada)
        state, new_buffer = rollout(settings, network_params,state, buffer, runpar)

        # Vetoriza a função GAE
        vmapped_gae = jax.vmap(
            partial(general_advantage_estimator, settings, network_params),
            in_axes=(0, 0, 0, 0)
        )

        # calcula as vantagens (usando a função vetorizada)
        advantages, returns = vmapped_gae(
            new_buffer.obs_buffer,
            new_buffer.reward_buffer,
            new_buffer.ptr,
            new_buffer.done_flag,
        )

        # normaliza para prevenir problemas com os gradientes, com as recompensas ruidosas
        advantages = mu.stdNormalize(advantages)
        returns = mu.stdNormalize(returns)

        #bloqueia o calculo de gradientes 
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)


        return state, new_buffer, advantages, returns
    
    def new_goal_step(carry, _):
        """This is the body of the scan, representing one full update."""
        runpar, optim_state, network_params, current_state = carry
        runpar = cast(RunningParameters, runpar)
        network_params = cast(NetworkParameters, network_params)

        # cria um buffer
        buffer = BatchedBuffer.init(settings)

        #atualiza os goals com base no estado atual
        new_goal_state = update_goal(current_state, runpar.progress.value, settings)

        #coleta os dados e atualiza os parâmetros correntes 
        state, buffer, advantages, returns = collect_rollouts(new_goal_state, buffer, runpar, network_params)
        mean_envs_success_rate = get_success_rate(state)
    

        # metricas tem shape (epochs, *metric_shape)
        network_params, optim_state, training_metrics = train_epochs(settings, network_params, optim_state, buffer, advantages, returns)
        runpar = runpar.update(buffer.obs_buffer, mean_envs_success_rate)
    
        newcarry = (runpar, optim_state, network_params, state)
        err_tol = settings.robot_shared_data.reward_config.err_tol.update(mean_envs_success_rate)
        return newcarry, (training_metrics, mean_envs_success_rate, buffer.reward_buffer, state["err"], err_tol)

    # loop principal de trainamento, executado por lax.scan
    runpar = RunningParameters.init((settings.network_settings.obs_size, ), settings.target_success)
    rng1, initial_state = create_initial_state(rng, runpar.progress.value, settings)

    # após  o scan, teremos o seguinte:
    # training_metrics[key].shape = (numberof_goals, epochs, *metric_shape)
    # mean_envs_success_rate.shape = (numberof_goals,)
    # rewards.shape = (numberof_goals, num_envs, rollout_steps +1)
    final_carry, (training_metrics, mean_envs_success_rate, rewards, err, err_tol) = jax.lax.scan(
        new_goal_step,
        (runpar, settings.optimizer_state, starting_network_params, initial_state),
        jnp.arange(settings.active_numberof_goals),
    )

    # shape das perdas é: (numberof_goals, epochs,). Para exibir no formato (epochs, )
    avg_loss = jnp.mean(training_metrics["loss"], axis=0)
    avg_entropy = jnp.mean(training_metrics["entropy"], axis=0)
    avg_grad_norm = jnp.mean(training_metrics["grad_norm"], axis=0)

    # shape de recompensas é: (numberof_goals, num_envs, rollout_steps +1)
    #jax.debug.print("rewards shape: {}", rewards.shape)

    total_rollout_reward = jnp.sum(rewards, axis = 2)  #soma as recompensas do rollout: (numberof_goals, num_envs)
    mean_rewards_vs_goals = jnp.mean(total_rollout_reward, axis = 1) #(numberof_goals, )

    #mean reward across timestamp
    total_goals_reward = jnp.sum(rewards, axis=0) # (num_envs, rollout_steps+1)
    mean_rewards_vs_timestamp = jnp.mean(total_goals_reward, axis=0) #(rollout_steps+1, )



    print(f"mean_envs_success_rate shape: {mean_envs_success_rate.shape}")
    #success_rate_around_goals = jnp.mean(mean_envs_success_rate, axis=1)
    #success_rate_around_cycles = jnp.mean(mean_envs_success_rate, axis=0)

    print(f"err shape: {err.shape}")

    # err.shape = (numberof_goals, num_envs,)
    avg_err = jnp.mean(err, axis=1)

  

    metrics = {
        "avg_loss": avg_loss,
        "avg_entropy": avg_entropy,
        "avg_gradnorm":avg_grad_norm,
        "mean_rewards_vs_goals": mean_rewards_vs_goals,
        "mean_rewards_vs_timestamp":mean_rewards_vs_timestamp,
        "success_rate":mean_envs_success_rate,
        "avg_err": avg_err,
        "err_tol":err_tol,
    }

    #final_carry = (runpar, optim_state, network_params)
    return final_carry, metrics


def create_initial_state(rng: jax.Array, progress, settings: TrainingSettings):
    rng, rng1 = jax.random.split(rng)

    # Inicializa os estados para os ambientes em paralelo
    # (num_envs, features_dim)
    num_envs = settings.num_envs
    batched_rng = jax.random.split(rng, num_envs)

    mjx_data = mjx.make_data(settings.robot_shared_data.mjx_model)
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0),
        mjx_data
    )

    vmapped_get_goal = jax.vmap(partial(get_goal, settings.robot_shared_data.range_config, progress))
    batched_rng, batched_goal = vmapped_get_goal(batched_rng)


    # Crie um estado temporário para rodar o pipeline
    temp_state = {
        "rng": batched_rng,
        "goal": batched_goal,
        "mjx_data": batched_mjx_data,
        "obs": jnp.zeros((num_envs, settings.network_settings.obs_size)), # placeholder
        "last_action": jnp.zeros((num_envs, settings.network_settings.action_size)),
        "err": jnp.ones((num_envs,)) * jnp.inf,
        "step": jnp.zeros((num_envs,)),
        "success_count": jnp.zeros((num_envs,)),
    }

    # Rode apenas o pipeline de observação para obter o estado REAL inicial
    # Isso garante que a primeira obs que o agente vê não seja zero
    runpar_init = RunningParameters.init((settings.network_settings.obs_size,), settings.target_success)
    
    def get_single_obs(s):
        # s é um único 'state' (scalars/unbatched arrays)
        # StateMonad.pure({}) inicia o pdata como um dict vazio
        pipe = obs_pipeline(settings.robot_shared_data, runpar_init.obs_stat, StateMonad.pure({}), settings.obs_noise_scale)
        _, out_data = pipe.run(s)
        return out_data["obs"]

    # vmap mapeia 'get_single_obs' sobre a primeira dimensão de todos os arrays no temp_state
    initial_obs = jax.vmap(get_single_obs)(temp_state)

    return rng1, {
        **temp_state,
        "obs": initial_obs, # Agora contém dados reais do MuJoCo
    }



