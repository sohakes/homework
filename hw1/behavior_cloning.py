import pickle
import tensorflow as tf
import numpy as np
import gym
import os
import tensorflow.contrib.eager as tfe

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_runs_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument("--max_timesteps", type=int, default=10000)
    args = parser.parse_args()

    batch_size = args.batch_size

    obs_act = None

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
        tf.keras.layers.Dense(100, activation="relu", input_shape=(obs_size,)),  # input shape required
        tf.keras.layers.Dense(40, activation="relu"),
        tf.keras.layers.Dense(act_size)
        ])

    y_ = model(observations_placeholder)
    y_p = model(observations_placeholder_p)
    loss = tf.losses.mean_squared_error(labels=actions_placeholder, predictions=y_) + tf.losses.get_regularization_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    import gym
    env = gym.make(args.envname)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        for epoch in range(args.num_epochs):
            total_loss = 0
            for i in range(0, observations.shape[0], batch_size):
                o = observations[i:i+batch_size]
                a = actions[i:i+batch_size]
                # Session execute optimizer and fetch values of loss
                _, l = sess.run([optimizer, loss],
                    feed_dict={observations_placeholder: o, actions_placeholder:a}) 
                #print(o, a, l, y_.eval(feed_dict={observations_placeholder: o}))
                #if i < 8:
                #    print(o, a, l, y_.eval(feed_dict={observations_placeholder: o}))
                total_loss = l
                #print('Epoch {0}: {1}'.format(i, total_loss/batch_size))
            print('Epoch {0}: {1}'.format(epoch, total_loss))
            #sess.run(iterator.initializer, feed_dict={observations_placeholder: observations,
            #                              actions_placeholder: actions})
            max_steps = args.max_timesteps or env.spec.timestep_limit
            if (epoch+1) % 20 == 0:
                returns = 0
                obst = env.reset()
                done = False
                steps = 0
                while not done:
                    #act = sess.run([actions_placeholder], feed_dict={observations_placeholder: obst[None,:]}) 
                    act = y_p.eval(feed_dict={observations_placeholder_p: obst[None,:]})
                    obst, r, done, _ = env.step(act)
                    env.render()
                    returns += r
                    steps += 1
                    if steps % 100 == 0:
                        print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

                print('returns', returns)

        


if __name__ == '__main__':
    main()
