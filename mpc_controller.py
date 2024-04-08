import numpy as np
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

import do_mpc
from casadi import *



class MPC_Controller:
    def __init__(self): 
        self.TAU = 1
        self.ZETA = 1
        self.OMEGA = 1
        self.MASS = 375
        self.L = 2
        self.ENGINEBREAKS = 50

        #MPC Setup

        self.PREDICTION_HORIZON = 50
        self.T_STEPS = 0.1
        self.ROBUST = 1
        self.STORE_SOLUTION = True

        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        #Setting state-variables

        x_pos = model.set_variable(var_type='_x', var_name='x_pos', shape=(1,1))
        y_pos = model.set_variable(var_type='_x', var_name='y_pos', shape=(1,1))
        psi = model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(1,1))

        #Setting variables for Force and steering angle

        F = model.set_variable(var_type='_x', var_name='F', shape=(1,1))
        delta = model.set_variable(var_type='_x', var_name='delta', shape=(1,1))
        delta_dot = model.set_variable(var_type='_x', var_name='delta_dot', shape=(1,1))


        #Setting reference variables

        x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(1,1))
        y_ref = model.set_variable(var_type='_tvp', var_name='y_ref', shape=(1,1))
        velocity_ref = model.set_variable(var_type='_tvp', var_name='velocity_ref', shape=(1,1))


        #Setting control input variablers

        u_1 = model.set_variable(var_type='_u', var_name='u_1')
        u_2 = model.set_variable(var_type='_u', var_name='u_2')

        #Setting constant variables for model



        #State-space model

        x_dot = v*np.cos(psi)
        y_dot = v*np.sin(psi)
        psi_dot = v/self.L * np.tan(delta)
        v_dot = F/self.MASS
        F_dot = 1/self.TAU * (-F + u_1) - self.ENGINEBREAKS*v
        delta_d = delta_dot
        delta_ddot = -2*self.ZETA*self.OMEGA * delta_dot - self.OMEGA**2 * delta + u_2 


        model.set_rhs('x_pos', x_dot)
        model.set_rhs('y_pos', y_dot)
        model.set_rhs('psi', psi_dot)
        model.set_rhs('v', v_dot)
        model.set_rhs('F', F_dot)
        model.set_rhs('delta', delta_d)
        model.set_rhs('delta_dot', delta_ddot)

        model.setup()

        N_STEPS = 1000

        xref_list = []
        yref_list = []
        vref_list = []
        for i in range(N_STEPS):
            rotasjon = (4*np.pi)/N_STEPS
            # Her generer vi input referanser, atm så er det en sirkel med radius *50* og *2* runder i vår simuleringstid
            r = 50
            xref_list.append(r*np.cos(i*rotasjon/4))
            yref_list.append(r*np.sin(i*rotasjon/4))
            vref_list.append(0)



        self.mpc = do_mpc.controller.MPC(model)


        setup_mpc = {
            'n_horizon': self.PREDICTION_HORIZON,
            't_step': self.T_STEPS,
            'n_robust': self.ROBUST,
            'store_full_solution': self.STORE_SOLUTION,
        }
        self.mpc.set_param(**setup_mpc)

        #Tuning parameters

        X_WEIGHT = 2
        Y_WEIGHT = 2
        PSI_WEIGHT = 0
        V_WEIGHT = 0
        F_WEIGHT = 0
        DELTA_WEIGHT = 0
        DELTA_DOT_WEIGHT = 0

        lterm = (X_WEIGHT * (x_pos-x_ref)**2 + 
                Y_WEIGHT * (y_pos-y_ref)**2 +  
                V_WEIGHT * (v-velocity_ref)**2)
        mterm = lterm*0   #This is the most scuffed way to say 0. I want it to be zero, as the mayer term focuses on final state, and there is no meaningfull final state

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        #R matrix
        U_1_WEIGHT = 0.1
        U_2_WEIGHT = 0.1


        self.mpc.set_rterm(
            u_1=U_1_WEIGHT,
            u_2=U_2_WEIGHT
        )


        #Input bounds
        THROTTLE_LOW = 0      
        THROTTLE_HIGH = 5000
        STEERING_LOW = -pi/4  
        STEERING_HIGH = pi/4

        #State bouinds
        V_LOW = 0                
        V_HIGH = 100

        DELTA_LOW = -pi/4         
        DELTA_HIGH = pi/4


        # Lower bounds on states:
        self.mpc.bounds['lower','_x', 'v'] = V_LOW
        self.mpc.bounds['lower','_x', 'delta'] = DELTA_LOW

        # Upper bounds on states:

        self.mpc.bounds['upper','_x', 'v'] = V_HIGH
        self.mpc.bounds['upper','_x', 'delta'] = DELTA_HIGH


        # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'u_1'] = THROTTLE_LOW
        self.mpc.bounds['lower','_u', 'u_2'] = STEERING_LOW
        # Lower bounds on inputs:
        self.mpc.bounds['upper','_u', 'u_1'] = THROTTLE_HIGH
        self.mpc.bounds['upper','_u', 'u_2'] = STEERING_HIGH


        def tvp_fun(t_now):
            
            # Calculate the current index based on the current time and the time step size.
            step = int(t_now // self.T_STEPS)

            # Initialize the tvp_template with the correct structure provided by do_mpc.
            tvp_template = self.mpc.get_tvp_template()

            # Loop over the prediction horizon.
            for k in range(self.PREDICTION_HORIZON):
                # Index for the reference lists. Use modulo to cycle the references if the end is reached.
                ref_index = (step + k) % len(xref_list)

                # Set the TVP for the current step in the horizon.
                tvp_template['_tvp', k, 'x_ref'] = xref_list[ref_index]
                tvp_template['_tvp', k, 'y_ref'] = yref_list[ref_index]
                tvp_template['_tvp', k, 'velocity_ref'] = vref_list[ref_index]

            return tvp_template


        tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()

        x0, y0, psi0, v0, F0, d0, dd0 = [0.,0.,0.,0.,0.,0.,0.]

        x_list = [x0]
        y_list = [y0]
        psi_list = [psi0] 
        v_list = [v0]
        F_list = [F0]
        delta_list = [d0]
        delta_dot_list = [dd0]

        x0_ = np.array([x_list[-1], y_list[-1], psi_list[-1], v_list[-1], F_list[-1], delta_list[-1], delta_dot_list[-1]]).reshape(-1,1)
        self.mpc.reset_history()
        self.mpc.set_initial_guess()
        self.mpc.x0 = x0_
        self.mpc.set_initial_guess()

    def predict_step(self, states):
        x, y, psi, v, F, delta, delta_dot = [states[0], states[1], states[2], states[3], states[4], states[5], states[6]]
        x0_ = np.array([x, y, psi, v, F, delta, delta_dot])
        u = self.mpc.make_step(x0_)
        return [u[0], u[1]]

        


