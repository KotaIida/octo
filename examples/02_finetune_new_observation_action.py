"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb
from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead, DiffusionActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.model.components.vit_encoders import SmallStem16
from octo.model.components.tokenizers import ImageTokenizer
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("task_name", None, "Task name for finetuning.")
flags.DEFINE_string("exp_name", None, "Experiment name for finetuning.")
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")
flags.DEFINE_integer("action_horizon", 50, "Length of action horizon.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("num_episodes", 1000, "Number of episodes")
flags.DEFINE_integer("num_steps", 600, "Number of steps per an episode.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

flags.DEFINE_bool(
    "augment",
    False,
    "Whether to use image augmentations",
)

flags.DEFINE_string("task", "language_conditioned", "image_conditioned or language_conditioned or multimodal")
flags.DEFINE_string("strategy", "uniform", "None or uniform or last for goal relabeling strategy")


workspace_augment_kwargs = dict(
    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
    random_brightness=[0.1],
    random_contrast=[0.9, 1.1],
    random_saturation=[0.9, 1.1],
    random_hue=[0.5],
    augment_order=[
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
)
wrist_augment_kwargs = dict(
    random_brightness=[0.1],
    random_contrast=[0.9, 1.1],
    random_saturation=[0.9, 1.1],
    random_hue=[0.5],
    augment_order=[
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
)


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(name=FLAGS.exp_name, project="Octo_Franka_Sim_Cushion")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    if "aloha_sim_cube" in FLAGS.task_name:
        image_obs_keys={"primary": "top"}
        resize_size={"primary": (256, 256)}
        action_dim = 14
    elif "mobile" in FLAGS.task_name:
        image_obs_keys={"primary": "angle", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}        
        resize_size={"primary": (256, 256), "left_wrist": (256, 256), "right_wrist": (256, 256)}
        action_dim = 14
        # image_obs_keys={"primary": "angle", "secondary": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}        
        # resize_size={"primary": (256, 256), "secondary": (256, 256), "left_wrist": (256, 256), "right_wrist": (256, 256)}
    elif "franka" in FLAGS.task_name:
        # image_obs_keys={"left_wrist": "left_wrist", "right_wrist": "right_wrist"}        
        # resize_size={"left_wrist": (256, 256), "right_wrist": (256, 256)}
        image_obs_keys={"primary": "angle", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}        
        resize_size={"primary": (256, 256), "left_wrist": (256, 256), "right_wrist": (256, 256)}
        if FLAGS.augment:
            image_augment_kwargs = {"primary": workspace_augment_kwargs, "left_wrist": wrist_augment_kwargs, "right_wrist": wrist_augment_kwargs}
        else:
            image_augment_kwargs = {}
        action_dim = 16
    else:
        image_obs_keys={"primary": "top", "secondary": "angle", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}        
        resize_size={"primary": (256, 256), "secondary": (256, 256), "left_wrist": (256, 256), "right_wrist": (256, 256)}
        action_dim = 14

    if FLAGS.task == "image_conditioned":
        language_key = None
        goal_relabeling_strategy = FLAGS.strategy
        keep_image_prob = 1.0
    elif FLAGS.task == "language_conditioned":
        language_key = "language_instruction"
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif FLAGS.task == "multimodal":
        language_key = "language_instruction"
        goal_relabeling_strategy = FLAGS.strategy
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=FLAGS.task_name,
            data_dir=FLAGS.data_dir,
            image_obs_keys=image_obs_keys,
            proprio_obs_key="state",
            language_key=language_key,
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=FLAGS.action_horizon,
            goal_relabeling_strategy=goal_relabeling_strategy
        ),
        frame_transform_kwargs=dict(
            resize_size=resize_size,
            image_augment_kwargs=image_augment_kwargs
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    if FLAGS.task == "image_conditioned":
        text_processor = None
    else:
        text_processor = pretrained_model.text_processor

    def process_batch(batch):
        if FLAGS.task != "image_conditioned":
            batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    if "wrist" in config["model"]["observation_tokenizers"].keys():
        del config["model"]["observation_tokenizers"]["wrist"]
    ###
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # config["model"]["observation_tokenizers"]["secondary"] = ModuleSpec.create(
    #     ImageTokenizer,
    #     obs_stack_keys=["image_secondary"],
    #     task_stack_keys=[],
    #     encoder=ModuleSpec.create(SmallStem16)
    # )
    # config["model"]["observation_tokenizers"]["left_wrist"] = ModuleSpec.create(
    #     ImageTokenizer,
    #     obs_stack_keys=["image_left_wrist"],
    #     task_stack_keys=[],
    #     encoder=ModuleSpec.create(SmallStem16)
    # )
    # config["model"]["observation_tokenizers"]["right_wrist"] = ModuleSpec.create(
    #     ImageTokenizer,
    #     obs_stack_keys=["image_right_wrist"],
    #     task_stack_keys=[],
    #     encoder=ModuleSpec.create(SmallStem16)
    # )

    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=FLAGS.action_horizon,
        action_dim=action_dim,
        readout_key="readout_action",
    )

    # config["model"]["heads"]["action"] = ModuleSpec.create(
    #     DiffusionActionHead,
    #     action_horizon=FLAGS.action_horizon,
    #     action_dim=action_dim,
    #     readout_key="readout_action",
    # )
    # config["model"]["heads"]["action"]["kwargs"]["action_dim"] = action_dim
    # config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = FLAGS.action_horizon
    # config["model"]["heads"]["action"]["kwargs"]["diffusion_steps"] = 1
    # config["model"]["heads"]["action"]["kwargs"]["loss_type"] = "l1"

    config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_kwargs"]["keep_image_prob"] = keep_image_prob
    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    total_steps = FLAGS.num_epochs * FLAGS.num_episodes * FLAGS.num_steps // FLAGS.batch_size
    for i in tqdm.tqdm(range(total_steps), total=total_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % 1000 == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
