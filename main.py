import torch
import torch.nn as nn
import torch.optim as optim
import rmoss_interfaces.msg
from rclpy.qos import QoSProfile
from rclpy.node import Node
import numpy as np
import rclpy    
from geometry_msgs.msg import Twist
import random
from actor import Actor
from critic import CriticRNN

import std_msgs.msg

class RLAgent(Node):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        super().__init__("rl_agent")
        self.actor = Actor(state_size, action_size)
        self.critic = CriticRNN(state_size, 64, 1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.red_health = 500
        self.blue_health = 500

        # Subscribe to the state topic
        qos_profile = QoSProfile(depth=10)
        self.red_state_subscriber = self.create_subscription(
            std_msgs.msg.Float32MultiArray,
            '/rl_agent_red/state',
            self.state_callback_red,
            qos_profile
        )
        self.blue_state_subscriber = self.create_subscription(
            std_msgs.msg.Float32MultiArray,
            '/rl_agent_blue/state',
            self.state_callback_blue,
            qos_profile
        )

        # Define the action publisher
        self.red_publisher = self.create_publisher(
            Twist,
            "/red_standard_robot1/cmd_vel",
            10,
        )
        self.blue_publisher = self.create_publisher(
            Twist,
            "/blue_standard_robot1/cmd_vel",
            10,
        )
        # Placeholder for the current state and action
        self.current_state_red = None
        self.current_action_red = None
        self.current_state_blue = None
        self.current_action_blue = None

    def red_won(self):
        
        if self.current_state_blue is not None:
            blue_health = self.current_state_blue.tolist()[0]
            return blue_health[0] <= 0
        else:
            return False
        

        
    
    def blue_won(self): 
        
        if self.current_state_red is not None:
            red_health = self.current_state_red.tolist()[0]
            return red_health[0] <= 0
        else:
            return False
    


    def state_callback_blue(self, msg):
        state = np.array(msg.data).reshape((1, -1))
        self.current_state_blue = state
        # Decide on the next action using the actor
        
        action = self.select_action(state)
        self.publish_action_blue(action)

        # Update the current action (for future reference during reward calculation)
        self.current_action_blue = action

    def state_callback_red(self, msg):
        print("Got a callback")
        state = np.array(msg.data).reshape((1, -1))
        self.current_state_red = state
        # Decide on the next action using the actor
        action = self.select_action(state)
        # Publish the action
        self.publish_action_red(action)

        # Update the current action (for future reference during reward calculation)
        self.current_action_red = action

    def select_action(self, state):
        print("State shape: ", state.shape)
        state = torch.FloatTensor(state)
        state = (state - state.mean()) / (state.std() + 1e-8)

        action_probs = self.actor(state)
        action = torch.clamp(action_probs, -1.0, 1.0)  # Ensure actions are within the valid range
        return action.detach().numpy()

    def publish_action_blue(self, action):
        twist_msg = Twist()

        twist_msg.linear.x = random.uniform(0, 1)
        twist_msg.linear.y = random.uniform(0, 1)
        twist_msg.linear.z = random.uniform(0, 1)
        twist_msg.angular.x = random.uniform(0, 1)
        twist_msg.angular.y = random.uniform(0, 1)
        twist_msg.angular.z = random.uniform(0, 1)      
        self.blue_publisher.publish(twist_msg)


    def publish_action_red(self, action):
        twist_msg = Twist()
        # print(action)
        print("Hi! published movememt")
        twist_msg.linear.x = random.uniform(0, 1)
        twist_msg.linear.y = random.uniform(0, 1)
        twist_msg.linear.z = random.uniform(0, 1)
        twist_msg.angular.x = random.uniform(0, 1)
        twist_msg.angular.y = random.uniform(0, 1)
        twist_msg.angular.z = random.uniform(0, 1)      
        self.red_publisher.publish(twist_msg)


    def update_policy(self, reward):
        if self.current_state_red is not None:
            print("Current state red exists")
        if self.current_action_red is not None:
            print("Current action red exists")
        if self.current_state_blue is not None:
            print("Current state blue exists")
        if self.current_action_blue is not None:
            print("Current action blue exists")
            
        if (
            self.current_state_red is not None and
            self.current_action_red is not None and
            self.current_state_blue is not None and
            self.current_action_blue is not None
        ):
            print("Working to update the policy!")
            state_red = torch.FloatTensor(self.current_state_red)
            action_red = torch.FloatTensor(self.current_action_red)

            state_blue = torch.FloatTensor(self.current_state_blue)
            action_blue = torch.FloatTensor(self.current_action_blue)

            reward_red = torch.FloatTensor([reward]) if self.red_won() else torch.FloatTensor([0.0])
            reward_blue = torch.FloatTensor([reward]) if self.blue_won() else torch.FloatTensor([0.0])

            # Update Critic for red
            with open("Testing.txt", "w") as f:
                f.write(str(state_red.tolist()))

                f.close()
            current_value_red = self.critic(state_red)
            next_value_red = 0.0  # Placeholder for the next state
            target_red = reward_red + self.gamma * next_value_red
            critic_loss_red = nn.MSELoss()(current_value_red, target_red)
            self.critic_optimizer.zero_grad()
            critic_loss_red.backward()
            self.critic_optimizer.step()

            # Update Actor for red
            advantage_red = target_red - current_value_red.detach()
            # print("Current Value Red:", current_value_red)
            # print("Target Red:", target_red)
            # print("Advantage Red:", advantage_red)

            actor_output_red = self.actor(state_red)
            # print("Actor Output Red:", actor_output_red)
            actor_loss_red = -torch.sum(torch.log(actor_output_red) * advantage_red)  
            # print("Actor Loss Red:", actor_loss_red)
            self.actor_optimizer.zero_grad()
            actor_loss_red.backward()
            print("Actor loss red successful!")
            self.actor_optimizer.step()

            # Update Critic for blue
            current_value_blue = self.critic(state_blue)
            next_value_blue = 0.0  # Placeholder for the next state
            target_blue = reward_blue + self.gamma * next_value_blue
            critic_loss_blue = nn.MSELoss()(current_value_blue, target_blue)
            self.critic_optimizer.zero_grad()
            critic_loss_blue.backward()
            self.critic_optimizer.step()

            # Update Actor for blue
            advantage_blue = target_blue - current_value_blue.detach()
            actor_loss_blue = -torch.sum(torch.log(self.actor(state_blue)) * advantage_blue)
            self.actor_optimizer.zero_grad()
            actor_loss_blue.backward()
            self.actor_optimizer.step()

            # Reset current states and actions
            self.current_state_red = None
            self.current_action_red = None
            self.current_state_blue = None
            self.current_action_blue = None



def main(args=None):
    rclpy.init()
    state_size = (64*64 * 3 + 12 + 400)
    action_size = 6
    node = RLAgent(state_size, action_size)
    print("Starting to train the RL agent")

    num_episodes = 100  # Set the number of episodes

    try:
        for episode in range(num_episodes):
            print("Episode", str(episode))
            for permitted_moves in range(300):
                print("Hi frnak")
                rclpy.spin_once(node)  # Spin once to handle callbacks
                

                if node.red_won() or node.blue_won():
                    print("Game over")
                    
                    # Calculate the reward based on the game outcome
                    reward = 1.0 if node.red_won() else -1.0
                    node.update_policy(reward)
                    break

    except KeyboardInterrupt:
        print("Interrupted ")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
