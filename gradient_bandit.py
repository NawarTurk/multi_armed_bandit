from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def softmax(x, tau=1):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def gradient_bandit(
    heroes: Heroes, 
    alpha: float, 
    use_baseline: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Gradient Bandit action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param alpha: The learning rate.
    :param use_baseline: Whether or not use avg return as baseline.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    h = np.array([0]*num_heroes, dtype=float)  # init h (the logits)   ther eis a update here somewher eand u need to update it
    rew_record = []                            # Rewards at each timestep
    avg_ret_record = []                        # Average reward up to each timestep
    tot_reg_record = []                        # Total regret up to each timestep
    opt_action_record = []                     # Percentage of optimal actions selected
    
    reward_bar = 0
    total_rewards = 0
    total_regret = 0

    ################################
    ######### WRITE YOUR CODE 
    ###############################
    # extract true_probability_list 
    true_probability_list = []
    for hero in heroes.heroes:
        true_probability_list.append(hero['true_success_probability'])

    optimal_reward = max(true_probability_list)
    optimal_hero_index = true_probability_list.index(optimal_reward)

    # Defnning necessary additional variable
    optimal_action_selection_count = 0 
    action_selection_counter = [0]*num_heroes # this holds how many times each action were selected

    ######### 

    for t in range(heroes.total_quests):
        ######### WRITE YOUR CODE HERE
        hero_index = None

        # 1. choose a hero based on the probabilities 
        probabilities = softmax(h)
        hero_index = np.random.choice(range(num_heroes), p=probabilities)

        # 2. Updating the action counter values 
        action_selection_counter[hero_index] += 1

        # 3. Calculating and appending the reward at step t
        reward = heroes.attempt_quest(hero_index)
        rew_record.append(reward)
        
        # 4. Calculating and appending the running reward average  reward_bar
        total_rewards += reward
        avg_ret = total_rewards/(t+1)
        avg_ret_record.append(avg_ret)

        # 5. updating the logits
        if use_baseline:
            reward_bar = avg_ret
        else:
            reward_bar = 0

        for index in range(num_heroes):
            if index == hero_index:
                h[index] += alpha*(reward - reward_bar)*(1-probabilities[index])
            else:
                h[index] -= alpha*(reward - reward_bar)*probabilities[index]

        # 6. Calculating and appending the total regret at each step
        total_regret = (t + 1) * optimal_reward - total_rewards  # k * q*(a*) - total_rewards up to time t
        tot_reg_record.append(total_regret)

        # 7. Calculating and appending the optimal actoion selection %
        # Calculating how many times the algo selected the right action
        if hero_index == optimal_hero_index:
            optimal_action_selection_count += 1
        optimal_action_selection_percentage = optimal_action_selection_count/(t+1)
        opt_action_record.append(optimal_action_selection_percentage)  

        ######### 
 
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various alpha values with baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=True)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })
    
    save_results_plots(results_list, plot_title="Gradient Bandits (with Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_with_baseline.pdf')

    # Test various alpha values without baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=False)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Gradient Bandits (without Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_without_baseline.pdf')
