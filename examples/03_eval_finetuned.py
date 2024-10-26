"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_aloha_checkpoint>
"""
from functools import partial
import sys
import os
import tqdm

from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb
import cv2

import time

sys.path.append("/workspaces/octo/act")

# keep this to register ALOHA sim env
from envs.aloha_sim_env import AlohaGymEnv  # noqa
from envs.goal_img_creator import GoalImgCreator, extract_img_from_hdf

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_name", None, "Environment name for Gym simulation"
)
flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_integer(
    "num_rollouts", 3, "Number of rollouts."
)
flags.DEFINE_string(
    "output_dir", None, "Name of test."
)
flags.DEFINE_bool(
    "quick",
    False,
    "Whether quick is included in language description.",
)
flags.DEFINE_bool(
    "careful",
    False,
    "Whether careful is included in language description.",
)
flags.DEFINE_string("task", "language_conditioned", "image_conditioned or language_conditioned or multimodal")
flags.DEFINE_integer(
    "pickup_num",
    1,
    "Number of cucumbers to pick up",
)

flags.DEFINE_integer(
    "max_step",
    None,
    "Maximum step of a episode",
)

flags.DEFINE_integer(
    "action_horizon",
    50,
    "Lengh of action horizon",
)
flags.DEFINE_string("exp_name", None, "Experiment name for finetuning.")



def main(_):
    # setup wandb for logging
    wandb.init(name=FLAGS.exp_name, project="Octo_Franka_Sim_Cushion")
    np.random.seed(10)

    if FLAGS.env_name == "aloha-sim-cube-v0":
        max_step = 400
        replay_image_key = "image_primary"
    elif "mobile" in FLAGS.env_name or "franka" in FLAGS.env_name:
        if "bimanual" in FLAGS.env_name:
            max_step = 800
        elif "cushion" in FLAGS.env_name:
            max_step = 600
        else:
            max_step = 720
        # replay_image_key = "image_left_wrist"
    else:
        max_step = 600
        replay_image_key = "image_secondary"
        if FLAGS.env_name == "aloha-sim-cucumber-v0" or FLAGS.env_name == "aloha-sim-couple-cucumber-v0":
            del_tgt_indices = [1]
            repl_index_pairs = [
                [0, 0]
            ]
        elif FLAGS.env_name == "aloha-sim-cube-v0" or FLAGS.env_name == "aloha-sim-couple-cube-v0":
            del_tgt_indices = [0]
            repl_index_pairs = [
                [0, 1]
            ]
    replay_image_keys = ["image_primary", "image_secondary", "image_tertiary", "image_quaternary", "image_left_wrist", "image_right_wrist"]

    if FLAGS.quick:
        max_step //= 2
    if FLAGS.careful:
        max_step *= 2
    if int(FLAGS.pickup_num) > 1:
        max_step *= FLAGS.pickup_num
    if FLAGS.max_step is not None:
        max_step = FLAGS.max_step

    if FLAGS.task != "language_conditioned":
        bg_hdf5_dir = '/workspaces/octo/examples/envs/background/'
        bg_hdf5_basename = "episode_0.hdf5"

        ctr_hdf5_dir = '/workspaces/octo/examples/envs/center/'
        ctr_cucumber_hdf5_basename = "episode_0.hdf5"
        ctr_cube_hdf5_basename = "episode_1.hdf5"

        hdf5_file = os.path.join(bg_hdf5_dir, bg_hdf5_basename)
        bg_image = extract_img_from_hdf(hdf5_file, 0)

        hdf5_file = os.path.join(ctr_hdf5_dir, ctr_cucumber_hdf5_basename)
        ctr_bucket_image = extract_img_from_hdf(hdf5_file, 0)

        hdf5_file = os.path.join(ctr_hdf5_dir, ctr_cucumber_hdf5_basename)
        ctr_cucumber_image = extract_img_from_hdf(hdf5_file, -1)

        hdf5_file = os.path.join(ctr_hdf5_dir, ctr_cube_hdf5_basename)
        ctr_cube_image = extract_img_from_hdf(hdf5_file, -1)        
        goal_img_creator = GoalImgCreator(ctr_bucket_image.copy(), bg_image.copy(), [ctr_cucumber_image.copy(), ctr_cube_image.copy()])
        goal_img_creator.setup([0, 2], [1], [[48], [47]])

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    env = gym.make(FLAGS.env_name)

    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=FLAGS.action_horizon)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # running rollouts
    episode_returns = []
    first_reward_steps = []
    for _ in tqdm.tqdm(range(FLAGS.num_rollouts)):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        if FLAGS.quick:
            language_instruction[0] = "quickly " + language_instruction[0]
        if FLAGS.careful:
            language_instruction[0] = "carefully " + language_instruction[0]

        if FLAGS.task == "language_conditioned":
            task = model.create_tasks(texts=language_instruction)
        elif FLAGS.task == "image_conditioned":
            base_img = env.vis_images[0]
            goal_image = goal_img_creator(base_img, del_tgt_indices, repl_index_pairs)
            goal_image = cv2.resize(goal_image, (256, 256))
            task = model.create_tasks(goals={"image_primary": goal_image[None]})
            del task["language_instruction"]
        else:
            base_img = env.vis_images[0]
            goal_image = goal_img_creator(base_img, del_tgt_indices, repl_index_pairs)
            goal_image = cv2.resize(goal_image, (256, 256))
            task = model.create_tasks(goals={"image_primary": goal_image[None]}, texts=language_instruction)

        # run rollout for 400 steps
        images = {}
        for replay_image_key in replay_image_keys:
            images[replay_image_key] = [info["images"][replay_image_key]]
        episode_return = 0.0
        rewards = []
        # action_time = []
        # step_time = []
        print ("Inference Start")
        while len(images[replay_image_key]) < max_step:
            sys.stdout.write(f"\r {len(images[replay_image_key])}/{max_step}")
            sys.stdout.flush()
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            # action_start = time.perf_counter()
            # del obs["image_primary"]
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0] # get only 1 batch which containes 50 actions
            # action_end = time.perf_counter()

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)

            # step_end = time.perf_counter()

            # action_time.append(action_end-action_start)
            # step_time.append(step_end-action_end)

            for replay_image_key in replay_image_keys:
                images[replay_image_key].extend([images_info[replay_image_key] for images_info in info["images"]])
            rewards += info["rewards"]

            episode_return += reward
            if done or trunc:
                break

        print(f"\nEpisode return: {episode_return}")
        episode_returns.append(episode_return)

        first_reward_step = np.where(rewards)[0][0] if sum(rewards) > 0 else max_step
        first_reward_steps.append(first_reward_step)
        # log rollout video to wandb -- subsample temporally 2x for faster logging
        if FLAGS.task == "language_conditioned":
            for replay_image_key in replay_image_keys:
                wandb.log(
                    {replay_image_key: wandb.Video(np.array(images[replay_image_key]).transpose(0, 3, 1, 2), fps=50, format="mp4")}
                )
        else:
            wandb.log(
                {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2), fps=50, format="mp4"),
                 "goal_img": wandb.Image(goal_image)}
            )

    episode_returns = np.array(episode_returns)
    success_num = (episode_returns>0).sum()
    success_rate = success_num / len(episode_returns)
    first_reward_steps = np.array(first_reward_steps)
    average_first_reward_step = first_reward_steps[first_reward_steps < max_step].mean()
    summary_str = f"Rollout Num: {FLAGS.num_rollouts}\n" + f"Success Num: {success_num}\n" + f"Success Rate: {success_rate:.3f}\n" + f"Average First Reward Step: {average_first_reward_step}\n"
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    with open(os.path.join(FLAGS.output_dir, "result.txt"), 'w') as f:        
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n")
        f.write(repr(first_reward_steps))

if __name__ == "__main__":
    app.run(main)
