from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots
import random

def eps_greedy(
    heroes: Heroes, 
    eps: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform epsilon-greedy action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param eps: The epsilon value for exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: The average of rewards up to step t. For example: If 
    we define `ret_T` = sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """
    
    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes    # Initial action values
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []     #what is rest here?              # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
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
        # 1. Implement the eps_greedy selection algorithm:
        hero_index = None
        if random.random() >= eps:  # exploit ...
            # max() always choose the first element in case of multiple max values
            # below is ensuring that we are breaking the tie with a random selection as stated in the theory
            max_value = max(values)
            max_value_indicies = []
            for i in range(len(values)):
                if values[i] == max_value:
                    max_value_indicies.append(i)
            hero_index = random.choice(max_value_indicies)  # argmax
        else:  # explore ...
            hero_index = random.randint(0, num_heroes-1) # random
            # note random.randint is exclusive in both ends

        # 2. Calculating and appending the reward at step t
        reward = heroes.attempt_quest(hero_index)
        rew_record.append(reward)

        # 3. Updating action values Qt(a) and the action counter
        # we perform incremental implementation method on the chosen action
        # new_estimate = old_estimate + step_size * (target - old_estimate)
        action_selection_counter[hero_index] += 1
        current_action_counter = action_selection_counter[hero_index]
        values[hero_index] += (reward - values[hero_index])/current_action_counter

        # 4. Calculating and appending the running reward average
        total_rewards += reward
        avg_ret = total_rewards/(t+1)
        avg_ret_record.append(avg_ret)

        # 5. Calculating and appending the total regret at each step
        total_regret = (t + 1) * optimal_reward - total_rewards  # k * q*(a*) - total_rewards up to time t
        tot_reg_record.append(total_regret)

        # 6. Calculating and appending the optimal actoion selection %
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


    # Test various epsilon values
    eps_values = [0.2, 0.1, 0.01, 0.]
    results_list = []
    for eps in eps_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=eps, init_value=0.0)
        
        results_list.append({
            'exp_name': f'eps={eps}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Epsilons', 
                       results_folder='results', pdf_name='epsilon_greedy_various_epsilons.pdf')


    # Test various initial value settings with eps=0.0
    init_values = [0.0, 0.5, 1]
    results_list = []
    for init_val in init_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=0.0, init_value=init_val)
        
        results_list.append({
            'exp_name': f'init_val={init_val}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })
    
    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Initial Values',
                       results_folder='results', pdf_name='epsilon_greedy_various_init_values.pdf')
          
          
          
          
          
            # optimal_action_reward = np.random.binomial(1, optimal_reward)
        # regret = optimal_acation_reward - reward
        # total_regret += regret
        # tot_reg_record.append(total_regret)