import numpy as np
import time
from datasets import Dataset
from decision_transformer.models.decision_transformer_ori import DecisionTransformer
import torch
from tqdm import tqdm
"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv

class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict


@torch.no_grad()
def run(path_prefix=None):

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    state_dim = 32
    act_dim = 1
    action_range = [-1,1]
    max_length = 24
    eval_context_length = 24
    MAX_EPISODE_LEN = 8760
    embed_dim = 128
    n_layer = 4
    n_head = 1
    activ_function = "relu"
    dropout = 0.1
    ordering = 0 
    init_temperature = 0.1
    target_entropy = -act_dim

    state_mean = np.array([-1.08321095e-04,  7.84999232e-05,  3.17183956e-01, -8.59969638e-05,
                        -4.53695420e-05, -5.53211511e-04, -1.05196154e-03, -1.53922700e-03,
                            5.21621946e-05,  5.21621946e-05, -1.61064364e-04, -3.74290923e-04,
                        -5.87517482e-04, -9.79144294e-03, -1.14841073e-02, -1.25403672e-02,
                        -1.40243621e-02, -1.23454423e-02,  3.23607074e-03,  4.98120029e-01,
                            5.01879971e-01,  7.82125052e-01,  2.17874948e-01,  5.00318910e-01,
                            4.99471781e-01,  7.14738364e-01,  2.85261636e-01,  8.70707525e-02,
                            4.15045460e-01,  3.53143340e-01, -3.03629688e-03,  1.19217509e-03])

    state_std = np.array([0.99993903, 1.00001967, 0.320913, 1.00004626, 1.00006979, 1.00118711,
                        1.00230277, 1.00341573, 1.00005474, 1.00005474, 1.00034996, 1.00064504,
                        1.00094, 1.02152108, 0.99659728, 0.99054938, 0.9894332, 0.98709768,
                        0.10652074, 0.46516913, 0.46516913, 0.402183, 0.402183, 0.45587039,
                        0.45589259, 0.43514003, 0.43514003, 0.06457884, 0.66863673, 0.42017393,
                        0.99251213, 0.99466163])
    
    if torch.cuda.is_available():
        state_mean = torch.from_numpy(state_mean).to(device)
        state_std = torch.from_numpy(state_std).to(device)


    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                action_range=action_range,
                max_length=max_length,
                eval_context_length=eval_context_length,
                max_ep_len=MAX_EPISODE_LEN,
                hidden_size=embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                n_ctx = 72,  # because K = 24
                n_inner=4 * embed_dim,
                activation_function=activ_function,
                n_positions=1024,
                resid_pdrop=dropout,
                attn_pdrop=dropout,
                stochastic_policy=True,
                ordering=ordering,
                init_temperature=init_temperature,
                target_entropy=target_entropy,
            ).to(device=device)

    num_envs = 5
    ep_return = [-6000,-6000,-6000,-6000,-6000]

    #path_prefix ="exp/2024.08.01/225112-default"
    with open(f"{path_prefix}/model_30.pt", "rb") as f:
        checkpoint = torch.load(f,map_location=torch.device(device))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)


    
    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    obs = agent.register_reset(obs_dict,return_actions = False)

    state = np.array(obs)
    state_input = state[:,:-1]
    states = (
            torch.from_numpy(state_input)
            .reshape(num_envs, state_dim)
            .to(device=device, dtype=torch.float32)
        ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
                num_envs, -1, 1
            )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
                num_envs, -1
            )
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    list_actions = []
    for t in tqdm(range(MAX_EPISODE_LEN)):
        actions = torch.cat(
                        [
                            actions,
                            torch.zeros((num_envs, act_dim), device=device).reshape(
                                num_envs, -1, act_dim
                            ),
                        ],
                        dim=1,)
        rewards = torch.cat(
                        [
                            rewards,
                            torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                        ],
                        dim=1,
                    )

        state_pred, action_pred, reward_pred = model.get_predictions(
                        (states.to(dtype=torch.float32)-state_mean)/state_std,
                        actions.to(dtype=torch.float32),
                        rewards.to(dtype=torch.float32),
                        target_return.to(dtype=torch.float32),
                        timesteps.to(dtype=torch.long),
                        num_envs=num_envs,
                    )
        action_prediction = action_pred.detach().cpu().numpy()
        list_actions.append(action_prediction)
        observations,reward,done,_ = env.step(action_prediction)
        #episode_return += reward
        
        obs = agent.compute_action(observations, return_actions = False)
        obs_trans = np.array(obs)
        state_input = obs_trans[:,:-1]
        state = (
                    torch.from_numpy(state_input).to(device=device).reshape(num_envs, -1, state_dim)
                )
        states = torch.cat([states,state],dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:,-1] = reward
        pred_return = target_return[:, -1] - reward
        target_return = torch.cat(
                    [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
                )
        
        timesteps = torch.cat(
                    [
                        timesteps,
                        torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                            num_envs, 1
                        )
                        * (t + 1),
                    ],
                    dim=1,
                )
        
        actions[:,-1] = action_pred
        
        #print(rewards)
        

        if done : 
            metrics_t = env.evaluate()
            #print(metrics_t)
            break

    list_act_array = np.array(list_actions)
    np.save(path_prefix+"/list_act_array.npy",list_act_array)

if __name__ == "__main__":
    path_prefix = ["exp/2024.08.01/224248-default","exp/2024.08.01/224457-default","exp/2024.08.01/224704-default"
                   ,"exp/2024.08.01/224908-default","exp/2024.08.01/225112-default"]

    for prefix_ in path_prefix:
        run(prefix_)
