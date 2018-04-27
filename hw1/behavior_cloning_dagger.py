import pickle
import tensorflow as tf
import numpy as np
import gym
import os
import tensorflow.contrib.eager as tfe
import tf_util
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_runs_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    batch_size = args.batch_size

    obs_act = None

    policy_fn = load_policy.load_policy(args.expert_policy_file)

    with open('expert_runs/'+args.expert_runs_file+ '.pkl', 'rb') as f:
        obs_act = pickle.load(f)

    observations = obs_act["observations"]
    actions = np.squeeze(obs_act["actions"], axis=1)
    print(observations.shape, actions.shape, actions[0])

    assert observations.shape[0] == actions.shape[0]

    obs_size = observations.shape[1]
    act_size = actions.shape[1]

    observations_placeholder = tf.placeholder('float32', (batch_size, obs_size ))
    actions_placeholder = tf.placeholder('float32', (batch_size, act_size))

    observations_placeholder_p = tf.placeholder('float32', (1, obs_size ))
    actions_placeholder_p = tf.placeholder('float32', (1, act_size))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(obs_size,),
            kernel_regularizer=tf.contrib.keras.regularizers.l2()),  # input shape required
        tf.keras.layers.Dense(64, activation="relu", input_shape=(obs_size,),
            kernel_regularizer=tf.contrib.keras.regularizers.l2()),  # input shape required
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.contrib.keras.regularizers.l2()),
        tf.keras.layers.Dense(act_size)
        ])

    y_ = model(observations_placeholder)
    y_p = model(observations_placeholder_p)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.losses.mean_squared_error(labels=actions_placeholder, predictions=y_) + 0.01*sum(reg_losses)

    mod_epoch = 600


    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    import gym
    env = gym.make(args.envname)
    with tf.Session() as sess:
        tf_util.initialize()
        sess.run(tf.global_variables_initializer()) 

        for epoch in range(args.num_epochs):
            total_loss = 0
            for ii in range(0, observations.shape[0], batch_size):
                i = ii
                if i + batch_size > observations.shape[0]:
                    i = observations.shape[0] - batch_size
                o = observations[i:i+batch_size]
                a = actions[i:i+batch_size]

                # Session execute optimizer and fetch values of loss
                _, l, areg_losses = sess.run([optimizer, loss, reg_losses],
                    feed_dict={observations_placeholder: o, actions_placeholder:a}) 
                total_loss = l

            print('Epoch {0}: {1}'.format(epoch, total_loss))

            max_steps = args.max_timesteps or env.spec.timestep_limit
            
            if epoch % mod_epoch == 0:
                mod_epoch-=50
                if mod_epoch <200:
                    mod_epoch=200
                #Our network rendering first
                returns2 = 0
                #5 rollouts, render the first
                our_obs = []
                for k in range(30):
                    obst = env.reset()
                    done = False
                    steps = 0
                    while not done:
                        act = y_p.eval(feed_dict={observations_placeholder_p: obst[None,:]})
                        our_obs.append(obst)
                        obst, r, done, _ = env.step(act)
                        if k == 0 and epoch % (mod_epoch*3) == 0:
                            env.render()
                        returns2 += r
                        steps += 1
                        if steps % 400 == 0:
                            print("%i/%i"%(steps, max_steps))
                        if steps >= max_steps:
                            break
                    #Just so it doesn't generate too much data
                    if len(our_obs) > observations.shape[0]*0.1: 
                        break
                print("returns:", returns2)
                returns = []
                expert_acts=[]
                #generating expert actions
                for our_obs_i in our_obs:
                    action = policy_fn(our_obs_i[None,:])
                    expert_acts.append(action)
                our_obs = np.array(our_obs)
                expert_acts = np.array(expert_acts)
                expert_acts = np.squeeze(expert_acts, axis=1)
                print(observations.shape, our_obs.shape, actions.shape, expert_acts.shape)
                observations=np.concatenate((observations, our_obs), axis=0)
                actions=np.concatenate((actions, expert_acts), axis=0)
                print(observations.shape, our_obs.shape, actions.shape, expert_acts.shape)
                        
       


if __name__ == '__main__':
    main()
