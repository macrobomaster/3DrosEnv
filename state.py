import rclpy
from rclpy.node import Node
import sensor_msgs.msg
import rmoss_interfaces.msg
from geometry_msgs.msg import Twist
import std_msgs.msg
import numpy as np


class State(Node):
    def __init__(self):
        super().__init__("state_control")
        self.last_ref_message_red = None
        self.last_camera_message_red = None
        self.last_lidar_message_red = None
        self.last_ref_message_blue = None
        self.last_camera_message_blue = None
        self.last_lidar_message_blue = None
        #red
        self.times = 0
        self.lidar_subscriber = self.create_subscription(
                sensor_msgs.msg.LaserScan,
                f"/red_standard_robot1/rplidar_a2/scan",
                self.lidar_callback_red,
                10,
            )
        self.referee_subscriber = self.create_subscription(
                rmoss_interfaces.msg.RobotStatus,
                f"/referee_system/red_standard_robot1/robot_status",
                self.referee_callback_red,
                10,
            )
        self.camera_subscriber = self.create_subscription(
                sensor_msgs.msg.Image,
                f"/red_standard_robot1/front_camera/image",
                self.camera_callback_red,
                10,
        )

        #blue
        self.lidar_subscriber = self.create_subscription(
                sensor_msgs.msg.LaserScan,
                f"/blue_standard_robot1/rplidar_a2/scan",
                self.lidar_callback_blue,
                10,
            )
        self.referee_subscriber = self.create_subscription(
                rmoss_interfaces.msg.RobotStatus,
                f"/referee_system/blue_standard_robot1/robot_status",
                self.referee_callback_blue,
                10,
            )
        self.camera_subscriber = self.create_subscription(
                sensor_msgs.msg.Image,
                f"/blue_standard_robot1/front_camera/image",
                self.camera_callback_blue,
                10,
        )



        # Create a publisher for the state
        self.state_publisher_red = self.create_publisher(std_msgs.msg.Float32MultiArray, f'/rl_agent_red/state', 10)
        self.state_publisher_blue = self.create_publisher(std_msgs.msg.Float32MultiArray, f'/rl_agent_blue/state', 10)

        self.create_timer(1, self.get_state_red)
        self.create_timer(1, self.get_state_blue)


    def lidar_callback_red(self, msg):
        temp = [5 if x == float('inf') or x == float("-inf") else x for x in msg.ranges]
        self.last_lidar_message_red = np.array(temp)

    def referee_callback_red(self, msg):
        self.times += 1
        # print(self.times)
        if self.times > 1000:
            print("Stopping the game!!!") # msg.remain_hp
            self.last_camera_message_red = np.array([0, msg.max_hp, msg.total_projectiles, msg.used_projectiles, msg.hit_projectiles, msg.gt_tf.translation.x, msg.gt_tf.translation.y, msg.gt_tf.translation.z, msg.gt_tf.rotation.x, msg.gt_tf.rotation.y, msg.gt_tf.rotation.z, msg.gt_tf.rotation.w])
        self.last_ref_message_red = np.array([0, msg.max_hp, msg.total_projectiles, msg.used_projectiles, msg.hit_projectiles, msg.gt_tf.translation.x, msg.gt_tf.translation.y, msg.gt_tf.translation.z, msg.gt_tf.rotation.x, msg.gt_tf.rotation.y, msg.gt_tf.rotation.z, msg.gt_tf.rotation.w])

    def camera_callback_red(self, msg):
        # print("Recieved red camera message")
        # if self.last_lidar_message_red is not None:
            # print("Lider message exists")
        # if self.referee_callback_red is not None:
            # print("Referee message exists")
        # print("Before resizing")
        # self.last_camera_message_red = np.array(msg.data).reshape((480, 640, 3))
        temp = np.array(msg.data).reshape((480, 640, 3))
        self.last_camera_message_red = np.resize(temp, (64, 64, 3))
        # print("Resized red camera messag  e")
    def lidar_callback_blue(self, msg):
        temp = [5 if x == float('inf') or x == float("-inf") else x for x in msg.ranges]
        total = 0
        total_5 = 0
        for val in temp:
            if val == float('inf') or val == float("-inf"):
                total += 1
            if val == 5:
                total_5 += 1

        
        print("Num of inf" , total)
        print("Num of 5", total_5)
        print("Hi")
        
        self.last_lidar_message_blue = np.array(temp)

    def referee_callback_blue(self, msg):
        self.last_ref_message_blue = np.array([msg.remain_hp, msg.max_hp, msg.total_projectiles, msg.used_projectiles, msg.hit_projectiles, msg.gt_tf.translation.x, msg.gt_tf.translation.y, msg.gt_tf.translation.z, msg.gt_tf.rotation.x, msg.gt_tf.rotation.y, msg.gt_tf.rotation.z, msg.gt_tf.rotation.w])

    def camera_callback_blue(self, msg):
        # print("Recieved blue camera message")
        # if self.last_lidar_message_blue is not None:
        #     print("Lider message exists")
        # if self.referee_callback_blue is not None:
        #     print("Referee message exists")
        temp = np.array(msg.data).reshape((480, 640, 3))
        self.last_camera_message_blue = np.resize(temp, (64, 64, 3))



    def get_state_red(self):
        # print("Working to publish state")
        # if self.last_ref_message_red is not None:
        #     print("Ref message exists")
        # if self.last_camera_message_red is not None:
        #     print("Camera message exists")
        # if self.last_lidar_message_red is not None:
        #     print("Lidar message exists")

        if self.last_ref_message_red is not None and self.last_camera_message_red is not None and self.last_lidar_message_red is not None:
            state = np.concatenate([self.last_ref_message_red, self.last_camera_message_red , self.last_lidar_message_red], axis=None)
            print("Shape of imagE: ", self.last_camera_message_red.shape)

            print("Shape of message: ", state.shape)
            state_msg = std_msgs.msg.Float32MultiArray()
            state_msg.data = state.tolist()  
            self.state_publisher_red.publish(state_msg)
            print("Published state")

        else:
            return None
        

    def get_state_blue(self):
        print("Working to publish blue state")
        if self.last_ref_message_blue is not None and self.last_camera_message_blue is not None and self.last_lidar_message_blue is not None:
            
            state = np.concatenate([self.last_ref_message_blue, self.last_camera_message_blue , self.last_lidar_message_blue], axis=None)
            print("Shape of blue message!: ", state.shape)


            state_msg = std_msgs.msg.Float32MultiArray()
            state_msg.data = state.tolist()  
            
            self.state_publisher_blue.publish(state_msg)

        else:
            return None








def main(args=None):
    rclpy.init()
    node = State()
    print("Starting to publish random data")

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("Interrupted ")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

