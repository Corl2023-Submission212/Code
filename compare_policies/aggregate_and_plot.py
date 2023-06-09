from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FLAGS = flags.FLAGS

import seaborn as sns
sns.set()
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

font_size = 20
small_fontsize =  16

flags.DEFINE_spaceseplist('logdirs', [], 
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.pdf', 
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')

def main(_):
    sns.color_palette()
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    print(FLAGS.logdirs)
    # Looping through each directory
    for logdir in FLAGS.logdirs:
        print(logdir)
        samples = []
        rewards = []
        # Collecting data from each seed
        for seed in range(FLAGS.seeds):
            logdir_ = Path(logdir) / f'seed{seed}'
            logdir_ = logdir_ / 'val'
            event_acc = EventAccumulator(str(logdir_))
            event_acc.Reload()
            _, step_nums, vals = zip(*event_acc.Scalars('test_loss'))
            samples.append(step_nums)
            rewards.append(vals)
        samples = np.array(samples)
        assert(np.all(samples == samples[:1,:]))
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, 0)
        std_rewards = np.std(rewards, 0)

        # Plotting mean and std
        ax.plot(samples[0,:], mean_rewards, label=logdir)
        ax.fill_between(samples[0,:], 
                        mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)

    # Modifying plot visuals
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(small_fontsize)
    ax.legend(loc=2, fontsize=font_size)
    ax.set_xlabel("Epochs", fontsize=font_size)
    ax.set_ylabel("Validation Loss", fontsize=font_size)
    ax.grid('major')
    fig.savefig(FLAGS.output_file_name, bbox_inches='tight', format='svg', dpi=1200)


if __name__ == '__main__':
    app.run(main)
