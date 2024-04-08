#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from rclpy.duration import Duration

from std_msgs.msg import Float32, Int64, Float64MultiArray

import time

import numpy as np




class Simulator(Node):
   def __init__(self):
      super().__init__('simulator')
      self.TAU = 1 # tidskonstant for 1 ordens lavpass for force
      self.ZETA = 1 # demping for delta
      self.OMEGA = 1 # udempet svingefrekvens for andre ordens filter delta
      self.LEN = 1 # lengde fra hjul til hjul
      self.M = 375 # masse på LW
      self.MOTORBREMS = 100 # [ns/m] ganget sammen med farten v utgjør dette en enkel tilmerming for motorbrems

      self.X0 = 0
      self.Y0 = 0
      self.PSI0 = 0
      self.V0 = 0
      self.F0 = 0
      self.DELTA0 = 0
      self.DELTA_D0 = 0
      self.dt = 0.01

      self.u1 = 0
      self.u3 = 0


      self.u_1_subscriber = self.create_subscription(Int64, 'lw_portenta_throttle', self.update_throttle_input, 10)
      # self.brake_sub_ = self.create_subscription(Int64, 'lw_portena_brake', self.update_brake_input, 10)
      self.steering_sub_ = self.create_subscription(Int64, 'lw_portenta_steering', self.update_steering_input, 10)

      self.simulate = self.create_publisher(Float64MultiArray, '/lw_states', 10)

      self.sim_timer = self.create_timer(0.01, self.simulate_step)
      self.update_timer = self.create_timer(0.02, self.update_states)


   def update_throttle_input(self, msg):
      self.u1 = msg.data

   def update_steering_input(self, msg):
      self.u3 = msg.data / 1000



   def simulate_step(self):
      x_d = self.V0 * np.cos(self.PSI0)
      y_d = self.V0 * np.sin(self.PSI0)
      psi_d = self.V0 / self.LEN * np.tan(self.DELTA0)
      v_d = self.F0 / self.M
      f_d = 1 / self.TAU * (-self.F0 + self.u1) - self.MOTORBREMS * self.V0
      delta_d = self.DELTA_D0
      delta_der_d = -2 * self.ZETA * self.OMEGA * self.DELTA_D0 - self.OMEGA**2 * self.DELTA0 + self.u3
      
      self.X0 += x_d * self.dt
      self.Y0 += y_d * self.dt
      self.PSI0 += psi_d * self.dt
      self.V0 += v_d * self.dt
      self.F0 += f_d * self.dt
      self.DELTA0 += delta_d * self.dt
      self.DELTA_D0 += delta_der_d * self.dt
      
   def update_states(self):
      states = Float64MultiArray()
      states.data = [self.X0, self.Y0, self.PSI0, self.V0]
      self.simulate.publish(states)






def main(args=None):

   # X0 = 0
   # Y0 = 0
   # PSI0 = 0
   # V0 = 0
   # F0 = 0
   # DELTA0 = 0
   # DELTA_D0 = 0

   rclpy.init(args=args)
   node = Simulator()
   try: 
      rclpy.spin(node)
   except KeyboardInterrupt:
      #legg inn en fiks for at vi stopper hvis keyboard interrupt
      pass
      # save_data_to_cvs(node.time_stamp_, node.u1_list_, node.u2_list_,node.u3_list_, node.velocity_list_, node.angle_list_)

   finally: 
      rclpy.shutdown()


if __name__ == '__main__':
   main()


      