from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def boltzmann_policy(x, tau):
    """ Returns an index sampled from the softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    
    ######### WRITE YOUR CODE HERE
    action_values = np.array(x)

    exp_values = np.exp(action_values/tau)

    exp_values_sum = np.sum(exp_values)

    probabilities = exp_values / exp_values_sum

    index = np.random.choice(len(action_values), p=probabilities)
    #########  

    return index


def boltzmann(
    heroes: Heroes, 
    tau: float = 0.1, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Boltzmann action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param tau: The temperature value (𝜏). 
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes    # Initial action values
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []                   # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0

    ##############################
    ######### WRITE YOUR CODE HERE
    ##############################
    # extract true_probability_list to determine:
    # * optimal reward, and
    # * optimal hero index
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

        # 1. implement boltzmann_policy to choose the an action
        hero_index = boltzmann_policy(values, tau)

        # 2. Updating the action counter values 
        action_selection_counter[hero_index] += 1

        # 3. Calculating and appending the reward at step t
        reward = heroes.attempt_quest(hero_index)
        rew_record.append(reward)

        # 4. Updating action values Qt(a)
        # we perform incremental implementation method on the chosen action
        # new_estimate = old_estimate + step_size * (target - old_estimate)
        values[hero_index] += (reward - values[hero_index])/action_selection_counter[hero_index]

        
        # 5. Calculating and appending the running reward average
        total_rewards += reward
        avg_ret = total_rewards/(t+1)
        avg_ret_record.append(avg_ret)

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

    # Test various tau values
    tau_values = [0.01, 0.1, 1, 10]
    results_list = []
    for tau in tau_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=boltzmann,
                                                                    tau=tau, init_value=0)
        
        results_list.append({
            "exp_name": f"tau={tau}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Boltzmann Experiment Results On Various Tau Values",
                       results_folder='results', pdf_name='boltzmann_various_tau_values.pdf')
