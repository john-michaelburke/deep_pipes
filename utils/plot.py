import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="darkgrid")

def plot_loss(csv_path):
    datur = pd.read_csv(csv_path)
    losses = pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_CLIP"],
            "loss": ["L_clip"]*datur["TOTAL_STEPS"].shape[0]
        }
    )
    losses = losses.append(pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_VHS"],
            "loss": ["L_vf"]*datur["TOTAL_STEPS"].shape[0]
        },
        
    ),ignore_index=True)
    losses = losses.append(pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_ENTROPY"],
            "loss": ["L_s"]*datur["TOTAL_STEPS"].shape[0]
        }
    ),ignore_index=True)
    
    sns.lineplot(x="Total Steps", y="Loss", hue="loss",
                data=losses)
    plt.title("Average Optimization Batch Loss During Training")
    plt.savefig('PPO_loss_plot.png')
    plt.clf()

def plot_reward(csv_path):
    datur = pd.read_csv(csv_path)
    df = pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Total Reward": datur["LAST_REW"].rolling(10).mean(),
        }
    )
    
    fig = sns.lineplot(x="Total Steps", y="Total Reward",
                data=df)
    ax = fig.axes
    roll_std = datur["LAST_REW"].rolling(10).std()
    lower_bound = df["Total Reward"] - roll_std
    upper_bound = df["Total Reward"] + roll_std
    ax.fill_between(df['Total Steps'], lower_bound.values, upper_bound.values, alpha=.3)

    plt.title("Windowed Average Total Reward (10 Games)")
    plt.savefig('PPO_reward_plot.png')
    plt.clf()

def plot_loss_and_reward(csv_path):
    datur = pd.read_csv(csv_path)
    _, ax = plt.subplots(2)
    ax1 = ax[0]
    ax2 = ax[1]
    losses = pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_CLIP"],
            "loss": ["L_clip"]*datur["TOTAL_STEPS"].shape[0]
        }
    )
    losses = losses.append(pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_VHS"],
            "loss": ["L_vf"]*datur["TOTAL_STEPS"].shape[0]
        },
        
    ),ignore_index=True)
    losses = losses.append(pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Loss": datur["LOSS_ENTROPY"],
            "loss": ["L_s"]*datur["TOTAL_STEPS"].shape[0]
        }
    ),ignore_index=True)
    
    sns.lineplot(ax=ax1, x="Total Steps", y="Loss", hue="loss",
                data=losses)
    ax1.title.set_text("Average Optimization Batch Loss During Training")
    df = pd.DataFrame(
        {
            "Total Steps": datur["TOTAL_STEPS"],
            "Total Reward": datur["LAST_REW"].rolling(10).mean(),
        }
    )
    
    fig = sns.lineplot(ax=ax2, x="Total Steps", y="Total Reward",
                data=df)
    # ax = fig.axes
    roll_std = datur["LAST_REW"].rolling(10).std()
    lower_bound = df["Total Reward"] - roll_std
    upper_bound = df["Total Reward"] + roll_std
    ax2.fill_between(df['Total Steps'], lower_bound.values, upper_bound.values, alpha=.3)
    ax2.title.set_text("Windowed Average Total Reward (10 Games)")
    plt.subplots_adjust(hspace=0.6)
    plt.savefig('PPO_loss_and_reward_plot.png')
    plt.clf()


    

def plot_test_reward(csv_path):
    datur = pd.read_csv(csv_path)
    df = pd.DataFrame(
        {
            "game": datur["GAME"],
            "total_reward": datur["LAST_REW"],
        }
    )
    
    fig = sns.lineplot(x="game", y="total_reward",
                data=df)

    plt.title("Total Reward with Best Model")
    plt.savefig('PPO_test_reward_plot.png')
    plt.clf()
