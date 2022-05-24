
from core import utils as utils
import numpy as np
import torch


# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):



    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    while True: #一定會執行
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]#identifier = id(0,1,2...), 0

        fitness = 0.0
        total_frame = 0
        state = env.reset()
        rollout_trajectory = []
        #state = utils.to_tensor(state)
        state = utils.to_tensor(state.reshape(1,-1))#20220514
        while True:  # unless done

            #if type == 'pg': action = net.noisy_action(state)
            if type == 'pg': action = net.noisy_action(state.reshape(1,-1))#20220514
            #else: action = net.clean_action(state)
            else: action = net.clean_action(state.reshape(1,-1))#20220514

            action = utils.to_numpy(action)
            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment

            #next_state = utils.to_tensor(next_state)
            next_state = utils.to_tensor(next_state.reshape(1,-1))#20220514
            fitness += reward
            # If storing transitions
            if store_data: #Skip for test set
                rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
                                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
                                           np.reshape(np.float32(np.array([float(done)])), (1, 1))])

            """驗證用
            if store_data: #Skip for test set
                rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
                                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
                                           np.reshape(np.float32(np.array([float(done)])), (1, 1)),
                                           info])
            """                                       
            state = next_state
            total_frame += 1

            # DONE FLAG IS Received
            if done:
                break
        #print('rollout_trajectory: ',rollout_trajectory)
        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory])
