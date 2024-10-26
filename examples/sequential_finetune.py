from absl import app, flags
import subprocess
import inflect

FLAGS = flags.FLAGS

flags.DEFINE_string("script", "/workspaces/octo/examples/02_finetune_new_observation_action.py", "Path to finetuning script.")
flags.DEFINE_integer("epoch_per_run", 1, "Number of finetuning epochs per one running.")


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

def main(_):
    total_run = FLAGS.num_epochs//FLAGS.epoch_per_run
    p = inflect.engine()
    for run_idx in range(total_run):
        previous_ordinal = p.number_to_words(p.ordinal(run_idx))
        current_ordinal = p.number_to_words(p.ordinal(run_idx+1))
        if run_idx == 0:
            pretraind_path = FLAGS.pretrained_path
            save_dir = FLAGS.save_dir + "_" + current_ordinal
            exp_name = FLAGS.exp_name + " [" + current_ordinal + "]"
        else:
            pretraind_path = save_dir
            save_dir = save_dir.replace(previous_ordinal, current_ordinal)
            exp_name = exp_name.replace(previous_ordinal, current_ordinal)

        commands = ['python3', FLAGS.script, "--exp_name", exp_name, "--task_name", FLAGS.task_name, "--batch_size", str(FLAGS.batch_size), 
                    "--num_steps", str(FLAGS.num_steps), "--num_epochs", str(FLAGS.epoch_per_run), "--num_episodes", str(FLAGS.num_episodes),
                    "--data_dir", FLAGS.data_dir, "--pretrained_path", pretraind_path, "--save_dir", save_dir,
                    "--action_horizon", str(FLAGS.action_horizon), "--task", FLAGS.task, "--strategy", FLAGS.strategy]
        
        if FLAGS.freeze_transformer:
            commands.append("--freeze_transformer")
        if FLAGS.augment:
            commands.append("--augment")

        result = subprocess.run(commands, stdout=None, stderr=None)
        
        print (f"\n\n\nRun index:{run_idx} Finished.")
        if result.returncode != 0:
            print(f"Error raised. Stop Execution.")
            break

if __name__ == "__main__":
    app.run(main)
