


**Containers + Files**

1. container + entrypoint.sh file is for running the actual simulation
2. container + entrypoint8.sh file is for running rviz2. 

**Topics**

1. ros/clicked_point
2. /goal_pose
3. /initialpose
4. /parameter_events
5. /red_standard_robot1/cmd_vel
6. /red_standard_robot1/front_camera/image
7. /red_standard_robot1/robot_base/chassis_cmd
8. /red_standard_robot1/robot_base/gimbal_cmd
9. /red_standard_robot1/robot_base/gimbal_state
10. /red_standard_robot1/robot_base/shoot_cmd
11. /red_standard_robot1/rplidar_a2/scan
12. /referee_system/red_standard_robot1/enable_power
13. /referee_system/red_standard_robot1/robot_status
14. /rosout
15. /tf
16. /tf_static


Please note that instead of just using red_standard_robot1, you can use any color robot, followed by the _standard_robot1

**Game State**

For the Reinforcement learning model, we are determining the game state from the state.py file. 

The way that the state is determined is based off the most recent ROS2 topic publishes.

The following is how the state is stored:
```
self.last_ref_message, self.last_camera_message, self.last_lider_message
```
As you can see, it takes the most previous referee message, which looks like this:

```
---
id: 0
level: 0
name: ''
remain_hp: 500
max_hp: 500
total_projectiles: 200
used_projectiles: 0
hit_projectiles: 0
gt_tf:
  translation:
    x: -3.5
    y: -2.0
    z: 0.08
  rotation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
---

```

It then also outputs the last camera and lidar message, which have been ommitted for brevity, however their dimensions are 480x640 and 400 respectively. 

It then outputs the current state to /rl_agent_{color}/state, so that it can be easily accessed by the RL agent. 
 


I am deciding to go with an Actor-Critic based model. The critic can be found in the critic.py file. I am temporarily using a very simple RNN based model to determine the reward given the state. Obviously we will have to make this a deeper model so that we can better determine the reward "function". 

I have yet to test is the current functionality works. 
