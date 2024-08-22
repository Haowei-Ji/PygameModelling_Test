"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""
from math import atan2

import pygame
import numpy as np
from pygmodw23 import support


class AgentBase(pygame.sprite.Sprite):
    """
    Generalized agent class with very basic functionalities shared across all children
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()
        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.orig_color = color
        self.color = self.orig_color
        self.selected_color = support.LIGHT_BLUE

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Boundary conditions
        # bounce_back: agents bouncing back from walls as particles
        # infinite: agents continue moving in both x and y direction and teleported to other side (torus)
        self.boundary = "infinite"

        # Non-initialisable private attributes
        self.velocity = 1.1  # agent absolute velocity
        self.v_max = 1  # maximum velocity of agent
        self.vx = 0  # vlocity in x direction
        self.vy = 0  # velocity in y direction

        # Time step
        self.dt = 0.06
        self.dv = 0
        self.dtheta = 0

        # Interaction
        self.is_moved_with_cursor = 0

        # Visualization parameters
        self.show_stats = False
        self.change_color_with_orientation = False

        # Initial Visualization of agent
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        #test for ellipse
        pygame.draw.ellipse(self.image,color, [radius-radius *2 , radius - radius, radius, radius*2])

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        self.color = support.calculate_color(self.orientation, self.velocity)

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        if self.change_color_with_orientation:
            self.change_color()
        else:
            self.color = self.orig_color

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        try:
            if self.is_moved_with_cursor:
                pygame.draw.circle(
                    self.image, self.selected_color, (self.radius, self.radius), self.radius
                )
            else:
                pygame.draw.circle(
                    self.image, self.color, (self.radius, self.radius), self.radius
                )


        except:
            self.color[3] = 0
            pygame.draw.circle(
                self.image, self.color, (self.radius, self.radius), self.radius
            )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def reflect_from_walls(self, boundary_condition):
        """reflecting agent from environment boundaries according to a desired x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will be changed such that the agent is
         reflected from these boundaries."""

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius

        if boundary_condition == "bounce_back":
            # Reflection from left wall
            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[0] - self.radius

                if np.pi / 2 <= self.orientation < np.pi:
                    self.orientation -= np.pi / 2
                elif np.pi <= self.orientation <= 3 * np.pi / 2:
                    self.orientation += np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from right wall
            if x > self.boundaries_x[1]:

                self.position[0] = self.boundaries_x[1] - self.radius - 1

                if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                    self.orientation -= np.pi / 2
                elif 0 <= self.orientation <= np.pi / 2:
                    self.orientation += np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from upper wall
            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[0] - self.radius

                if np.pi / 2 <= self.orientation <= np.pi:
                    self.orientation += np.pi / 2
                elif 0 <= self.orientation < np.pi / 2:
                    self.orientation -= np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from lower wall
            if y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[1] - self.radius - 1
                if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                    self.orientation += np.pi / 2
                elif np.pi <= self.orientation < 3 * np.pi / 2:
                    self.orientation -= np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

        elif boundary_condition == "infinite":

            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[1] - self.radius
            elif x > self.boundaries_x[1]:
                self.position[0] = self.boundaries_x[0] + self.radius

            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[1] - self.radius
            elif y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[0] + self.radius

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def prove_velocity(self):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if np.abs(self.velocity) > self.v_max:
            # stopping agent if too fast during exploration
            self.velocity = self.v_max

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all other agents in the environment.
        """
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # # updating agent's state variables according to calculated vel and theta
            self.orientation += self.dt * self.dtheta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += self.dt * self.dv
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.vx = self.velocity * np.cos(self.orientation)
            self.vy = self.velocity * np.sin(self.orientation)
            self.position[0] += self.vx
            self.position[1] -= self.vy

            # boundary conditions if applicable
            self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.draw_update()


class AgentBrownianSelfPropelled(AgentBase):
    """
    Agent class realizing simple Brownian particle model.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad,
                 noise_type="normal", noise_params=(0, 1), v_max=1):
        """
        Initalization method of main agent class of the simulations

        Base parameters:

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels

        Specialized parameters for Brownian model:

        :param noise_type: type of noise to be used for Brownian motion. Can be "normal" or "uniform"
        :param noise_params: parameters of the noise distribution as (mean, std) for normal and (min, max) for uniform
        :param v_max: maximum velocity of the agent
        """
        # Initializing supercalss (Base Agent)
        super().__init__(id, radius, position, orientation, env_size, color, window_pad)

        # Setting up Brownian motion parameters
        self.dtheta = 0
        self.noise_type = noise_type
        self.noise_params = noise_params

        self.v_max = v_max

    def update_forces(self, agents):
        """Update rule for a single agent. This method is called by the environment to update the agent state.
        Calculating movement of the agent according to Brownian motion model and updating the agent state."""
        # Defining callback function for noise according to noise type
        if self.noise_type == "normal":
            noise = np.random.normal(self.noise_params[0], self.noise_params[1])
        elif self.noise_type == "uniform":
            noise = np.random.uniform(self.noise_params[0], self.noise_params[1])
        # Calculating new orientation
        self.dtheta = noise

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all other agents in the environment.
        """
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # # updating agent's state variables according to calculated vel and theta
            self.orientation += self.dtheta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            # fixed velocity on maximum speed
            self.velocity = self.v_max

            # updating agent's position
            self.vx = self.velocity * np.cos(self.orientation)
            self.vy = self.velocity * np.sin(self.orientation)
            self.position[0] += self.vx
            self.position[1] -= self.vy

            # boundary conditions if applicable
            self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.draw_update()

class AgentBrownian(AgentBase):
    """
    Agent class realizing simple Brownian particle model.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad,
                 noise_params_th=(0, 1), noise_params_v=(0, 1), gamma=0.1):
        """
        Initalization method of main agent class of the simulations

        Base parameters:

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels

        Specialized parameters for Brownian model:

        :param noise_type: type of noise to be used for Brownian motion. Can be "normal" or "uniform"
        :param noise_params_th: parameters of the noise distribution as (mean, std) for orientation
        :param noise_params_v: parameters of the noise distribution as (mean, std) for velocity
        :param gamma: slowing coefficient
        """
        # Initializing supercalss (Base Agent)
        super().__init__(id, radius, position, orientation, env_size, color, window_pad)

        # Setting up Brownian motion parameters
        self.dtheta = 0
        self.dvel = 0
        self.noise_params_th = noise_params_th
        self.noise_params_v = noise_params_v
        self.gamma = gamma

    def update_forces(self, agents):
        """Update rule for a single agent. This method is called by the environment to update the agent state.
        Calculating movement of the agent according to Brownian motion model and updating the agent state."""
        # Calculating new orientation
        self.dtheta = np.random.normal(self.noise_params_th[0], self.noise_params_th[1])
        self.dvel = np.random.normal(self.noise_params_v[0], self.noise_params_v[1])

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all other agents in the environment.
        """
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # # updating agent's state variables according to calculated vel and theta
            self.orientation += self.dtheta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            # fixed velocity on maximum speed
            self.velocity += (-self.gamma * self.velocity) + self.dvel

            # updating agent's position
            self.vx = self.velocity * np.cos(self.orientation)
            self.vy = self.velocity * np.sin(self.orientation)
            self.position[0] += self.vx
            self.position[1] -= self.vy

            # boundary conditions if applicable
            self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.draw_update()


class Agent(AgentBase):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Base Agent)
        super().__init__(id, radius, position, orientation, env_size, color, window_pad)

        # Specifying parameters for this special 3-zone agent
        # Interaction strength
        # Attraction
        self.s_att = 0.02
        # Repulsion
        self.s_rep = 5
        # Alignment
        self.s_alg = 8

        # Interaction ranges (Zones)
        # Attraction
        self.steepness_att = -0.5
        self.r_att = 250
        # Repulsion
        self.steepness_rep = -0.5
        self.r_rep = 50
        # Alignment
        self.steepness_alg = -0.5
        self.r_alg = 150

        # Noise
        self.noise_sig = 0.1

    def update_forces(self, agents):
        """Updateing overall social forces on agent according to velocity and distance of others"""
        # CALCULATING change in velocity and orientation in the current timestep
        # vel, theta = support.random_walk()
        # center point
        v1_s_x = self.position[0] + self.radius
        v1_s_y = self.position[1] + self.radius

        # point on agent's edge circle according to it's orientation
        v1_e_x = self.position[0] + (1 + np.cos(self.orientation)) * self.radius
        v1_e_y = self.position[1] + (1 - np.sin(self.orientation)) * self.radius

        # vector between center and edge according to orientation
        v1_x = v1_e_x - v1_s_x
        v1_y = v1_e_y - v1_s_y

        heading_vec = np.array([v1_x, v1_y])

        # CALCULATING attraction force with all agents:
        vec_attr_total = np.zeros(2)
        vec_rep_total = np.zeros(2)
        vec_alg_total = np.zeros(2)
        for ag in agents:
            if ag.id != self.id:
                # Distance between focal agent and given pair
                ag_pos_x = ag.position[0] + ag.radius
                ag_pos_y = ag.position[1] + ag.radius
                s_pos_x = self.position[0] + self.radius
                s_pos_y = self.position[1] + self.radius
                if self.boundary == "bounce_back":
                    distvec = np.array([ag_pos_x - s_pos_x, ag_pos_y - s_pos_y])
                elif self.boundary == "infinite":
                    distvec = support.distance_infinite(np.array([s_pos_x, s_pos_y]),
                                                        np.array([ag_pos_x, ag_pos_y]))

                # Difference between velocity between given agents
                s_vel = np.array([self.velocity * np.cos(self.orientation), - self.velocity * np.sin(self.orientation)])
                ag_vel = np.array([ag.velocity * np.cos(ag.orientation), - ag.velocity * np.sin(ag.orientation)])
                dvel = ag_vel - s_vel

                # Calculating interaction forces
                vec_attr_total += support.CalcSingleAttForce(self.r_att, self.steepness_att, distvec)
                vec_rep_total += support.CalcSingleRepForce(self.r_rep, self.steepness_rep, distvec)
                vec_alg_total += support.CalcSingleAlgForce(self.r_alg, self.steepness_alg, distvec, dvel)

        force_total = self.s_att * vec_attr_total - self.s_rep * vec_rep_total + self.s_alg * vec_alg_total

        vel = self.v_max * np.linalg.norm(force_total)
        closed_angle = support.angle_between(heading_vec, force_total)
        closed_angle = (closed_angle % (2 * np.pi))
        # at this point closed angle between 0 and 2pi, but we need it between -pi and pi
        # we also need to take our orientation convention into consideration to recalculate
        # theta=0 is pointing to the right
        if not np.isnan(closed_angle):
            if 0 < closed_angle < np.pi:
                theta = -closed_angle
            else:
                theta = 2 * np.pi - closed_angle
        else:
            theta = 0

        # Adding directional noise
        if self.noise_sig > 0.0:
            noiseP = np.random.normal(0.0, self.noise_sig, size=1)
            theta += noiseP[0]

        self.dtheta = theta
        self.dv = vel


class AgentSIR(AgentBrownianSelfPropelled):
    def __init__(self, id, radius, position, orientation, env_size, color, window_pad,
                 noise_type="normal", noise_params=(0, 1), v_max=1,
                 infection_prob=0.5, recovery_prob=0.2, death_prob=0.2, infection_radius=30):
        """
        Initalization method of main agent class of the simulations

        Base parameters:

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels

        Specialized parameters for Brownian model:

        :param noise_type: type of noise to be used for Brownian motion. Can be "normal" or "uniform"
        :param noise_params: parameters of the noise distribution as (mean, std) for normal and (min, max) for uniform
        :param v_max: maximum velocity of the agent

        Specialized parameters for SIR model:

        :param infection_prob: probability of infection when in contact with infected agent
        :param recovery_prob: probability of recovery when infected
        :param death_prob: probability of death when infected
        :param infection_radius: radius of infection
        """
        # Initializing superclass (Base Agent)
        super().__init__(id, radius, position, orientation, env_size, color, window_pad, noise_type, noise_params,
                         v_max)
        self.state = "S"
        self.infection_prob = infection_prob
        self.death_prob = death_prob
        self.recovery_prob = recovery_prob
        self.infection_radius = infection_radius

    def change_color(self):
        """Changing color of agent according to the agent's state"""
        if self.state == "S":
            self.orig_color = support.BLUE
        elif self.state == "I":
            self.orig_color = support.RED
        elif self.state == "R":
            self.orig_color = support.GREEN
        elif self.state == "D":
            self.orig_color = support.GREY

    def update_forces(self, agents):
        """Overriding update force to include state transitions of SIR model next to brownian motion"""
        # Updating motion forces
        super().update_forces(agents)
        # Updating state
        # getting infectious agents in contact with focal agent
        if self.state == "S":
            infectious_agents = [ag for ag in agents if ag.state == "I" and ag.id != self.id and self.is_in_contact(ag)]
            for infag in infectious_agents:
                if np.random.rand() < self.infection_prob:
                    self.state = "I"
                    break
        elif self.state == "I":
            if np.random.rand() < self.death_prob:
                self.state = "D"
            elif np.random.rand() < self.recovery_prob:
                self.state = "R"
        if self.state == "D":
            self.v_max = 0
            self.dtheta = 0

    def is_in_contact(self, ag):
        """Checking if agent is in contact with another agent"""
        dist = np.linalg.norm(np.array(self.position) - np.array(ag.position))
        if dist < self.radius + self.infection_radius:
            return True
        else:
            return False

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all other agents in the environment.
        """
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # # updating agent's state variables according to calculated vel and theta
            self.orientation += self.dtheta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            # fixed velocity on maximum speed
            self.velocity = self.v_max

            # updating agent's position
            self.vx = self.velocity * np.cos(self.orientation)
            self.vy = self.velocity * np.sin(self.orientation)
            self.position[0] += self.vx
            self.position[1] -= self.vy

            # boundary conditions if applicable
            self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.change_color()
        self.draw_update()