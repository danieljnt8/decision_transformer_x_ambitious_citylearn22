"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym

import torch
import numpy as np

from datasets import load_from_disk
import datasets
from datasets import Dataset
import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
#from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer_ori import DecisionTransformer
#from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
#from evaluation_citylearn import create_eval_episodes_fn_ori,evaluate_episode_rtg_ori,create_test_episodes_fn_ori,test_episode_rtg_ori
from trainer import SequenceTrainer, SequenceTrainerCustom, SequenceTrainerOri
from logger import Logger
import pandas as pd

from utils_.helpers import *


from utils_.variant_dict import variant

MAX_EPISODE_LEN = 8760


def update_loss_csv(iter_value, loss, filename='loss_per_epoch.csv',type_name="Epoch"):
    # Try to read the existing CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # If file does not exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[type_name, 'Loss'])
    
    # Append the new data to the DataFrame
    new_row = {type_name: iter_value, 'Loss': loss}
    df = df.append(new_row, ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)


class Experiment:
    def __init__(self, variant,dataset_path):
        

        """
        env = CityLearnEnv(schema="citylearn_challenge_2022_phase_2")
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3WrapperCustom(env)
        """

        #self.state_dim, self.act_dim, self.action_range = self._get_env_spec(env)
        self.state_dim = 32
        self.act_dim = 1
        self.action_range= [-1,1]

        
        self.initial_trajectories = self._get_initial_trajectories(dataset_path=dataset_path)



        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            self.initial_trajectories
        )

        print("state_mean "+str(self.state_mean))
        print("state_std "+ str(self.state_std ))

        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant["device"] #variant.get("device", "cuda:0")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_ctx = 72,  # because K = 24
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

     
        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 
        self.logger = Logger(variant)
    
    def _get_initial_trajectories(self,dataset_path):
        dataset = load_from_disk(dataset_path)
        dataset,_ = segment_v2(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        trajectories = datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()})

        return trajectories
   
    def _get_env_spec(self,env):
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
                -1,#float(env.action_space.low.min()) ,
                1#float(env.action_space.high.max()) ,
            ]
        return state_dim,act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False,iteration=0):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "args": self.variant,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
        }

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")
        else:
            with open(f"{path_prefix}/model_{iteration}.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"\nModel saved at {path_prefix}/model_{iteration}.pt")

       
    def _save_model_online_tuning(self, path_prefix, iter):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model_iter_{iter}.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model_iter_{iter}.pt")

        

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self,trajectories):
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(np.array(path["rewards"]).sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

            # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: city_learn")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        print(sorted_inds)
        #print(trajectories[1])
        for ii in sorted_inds:
            print(ii)
        #print(trajectories[0].keys())
        trajectories = [trajectories[int(ii)] for ii in sorted_inds]

        for trajectory in trajectories:
            for key in trajectory.keys():
                trajectory[key] = np.array(trajectory[key])


        return trajectories, state_mean, state_std



    def pretrain(self, schema_eval = "citylearn_challenge_2022_phase_1"):
        print("\n\n\n*** Pretrain ***")

        """

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        """
        """
        train_fn = create_eval_episodes_fn_ori(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema = "citylearn_challenge_2022_phase_1")
        eval_fn = create_eval_episodes_fn_ori(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema = "citylearn_challenge_2022_phase_2")
        test_fn = create_test_episodes_fn_ori(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema = "citylearn_challenge_2022_phase_3")
        """
        
        trainer = SequenceTrainerOri(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

       
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                dataloader=dataloader,
            )

            update_loss_csv(self.pretrain_iter,train_outputs["training/train_loss_mean"],filename=self.logger.log_path+'/loss_per_epoch_pretrain.csv'
                            ,type_name="Iteration")


            """
            
            train_outputs, train_reward, df_train,data_interaction_train,df_ts_train= self.evaluate(train_fn)
            eval_outputs, eval_reward, df_evaluate,data_interaction_eval,df_ts_eval = self.evaluate(eval_fn)
            #train_outputs, train_reward, df_train,data_interaction_train,df_ts_train= self.evaluate(train_fn)

            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )
            """
            if self.pretrain_iter % 10 == 0 : 
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                    iteration=self.pretrain_iter
                )

            if self.pretrain_iter - 1 == self.variant["max_pretrain_iters"]:
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                    iteration = self.pretrain_iter
                )

            
            #df_train.to_csv(self.logger.log_path+f"/train_pretrain_ori_iter_{self.pretrain_iter}.csv")
            #df_evaluate.to_csv(self.logger.log_path+f"/eval_pretrain_ori_iter_{self.pretrain_iter}.csv")

            


            
            
            """
            if self.pretrain_iter == self.variant["max_pretrain_iters"] - 1:
                df_ts_train.to_csv(self.logger.log_path+f"/train_TS_final.csv")
                df_ts_eval.to_csv(self.logger.log_path+f"/eval_TS_final.csv")


                data_path_train = self.logger.log_path+f"/data_interactions_train.pkl"
                Dataset.from_dict({k: [s[k] for s in data_interaction_train] for k in data_interaction_train[0].keys()}).save_to_disk(data_path_train)

                data_path_eval = self.logger.log_path+f"/data_interactions_eval.pkl"
                Dataset.from_dict({k: [s[k] for s in data_interaction_eval] for k in data_interaction_eval[0].keys()}).save_to_disk(data_path_eval)

                test_outputs, test_reward, df_test,data_interaction_test,df_ts_test = self.evaluate(test_fn)

                data_path_test = self.logger.log_path+f"/data_interactions_test.pkl"
                Dataset.from_dict({k: [s[k] for s in data_interaction_test] for k in data_interaction_test[0].keys()}).save_to_disk(data_path_test)

                df_test.to_csv(self.logger.log_path+f"/test_final.csv")
                df_ts_test.to_csv(self.logger.log_path+f"/test_TS_final.csv")
            """

            self.pretrain_iter += 1
        


    
    
    def evaluate(self,eval_fn):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        
        o,df_evaluate,data_interaction,df_ts = eval_fn(self.model)
        outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return"]
        return outputs, eval_reward,df_evaluate,data_interaction,df_ts

  
        
    def __call__(self):

        #utils.set_seed_everywhere(args.seed)

       

        
       

        print("\n\nMaking Eval Env.....")
        
        eval_env_schema = "citylearn_challenge_2022_phase_1"

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(schema_eval=eval_env_schema)


        #eval_envs.close()

def run_experiment(seed):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=seed)
    #parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=20)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=-6000)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=200)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=100) #30 #100

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=100) #10
    parser.add_argument("--online_rtg", type=int, default=-9000)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=100)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=20) #20
    parser.add_argument("--eval_interval", type=int, default=5)

    # environment options
    parser.add_argument("--device", type=str, default="cuda") ##cuda 
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp_4")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args), dataset_path="data_interactions/winner_dataset_phase_1.pkl")
   
    print("=" * 50)
    experiment()

if __name__ == "__main__":
    seeds = [53728, 12345, 67890, 54321, 98765]
    for seed in seeds:
        run_experiment(seed)

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53728)
    #parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=-9000)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=1000) #30

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=100) #10
    parser.add_argument("--online_rtg", type=int, default=-9000)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=12)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=20) #20
    parser.add_argument("--eval_interval", type=int, default=5)

    # environment options
    parser.add_argument("--device", type=str, default="cuda:0") ##cuda 
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args),dataset_path="data_interactions/RBCAgent1/model_RBCAgent1_timesteps_8760_rf_CombinedReward_phase_2_8760.pkl")
   
    print("=" * 50)
    experiment()
"""