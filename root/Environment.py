from pyrep import PyRep
from os.path import dirname, join, abspath
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np, cv2 as cv
from PIL import Image as I
import math
from Constants import *


class Environment(object):

    def __init__(self, not_render):
        super(Environment, self).__init__()
        self.controller = PyRep()
        scene_file = join(dirname(abspath(__file__)),'scene/scene_reinforcement_learning_env.ttt')
        self.controller.launch(scene_file, headless=not_render)

        self.actuator = Panda()
        self.actuator.set_control_loop_enabled(False)
        self.actuator.set_motor_locked_at_zero_velocity(True)

        self.top_camera = VisionSensor('Vision_TOP')
        self.side_camera = VisionSensor('Vision_SIDE')
        self.front_camera = VisionSensor('Vision_FRONT')

        self.target = Shape('target')
        self.target_initial_pos = self.target.get_position()
        self.start_sim()
        self.actuator_tip = self.actuator.get_tip()
        self.actuator_initial_position = self.actuator.get_joint_positions()

        self.POS_MIN = [0.8, -0.2, 1.0]
        self.POS_MAX = [1.0, 0.2, 1.4]

    def get_image(self, camera):
        view = (camera.capture_rgb()*255).round().astype(np.uint8)
        view = np.asarray(I.fromarray(view))
        return cv.cvtColor(view, cv.COLOR_BGR2GRAY)

    def get_state(self):
        positions = self.actuator.get_joint_positions()
        velocities = self.actuator.get_joint_velocities()
        target_position = self.target.get_position()
        images = (self.get_image(self.top_camera), self.get_image(self.side_camera), self.get_image(self.front_camera))
        return (positions, velocities, target_position, images)


    def done(self):

        done = [self.inside_range(  self.target.get_position()[i] - RANGE_DISCOUNT,
                                    self.target.get_position()[i] + RANGE_DISCOUNT,
                                    self.actuator.get_tip().get_position()[i]) for i in range(3)]

        return done[0] == True and (done[0]==done[1] and done[0]==done[2])

    def inside_range(self, min, max, x):
        return True if (min <= x and x <= max) else False

    def reset_scene(self):
        #new_target_pos = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        self.target.set_position(self.target_initial_pos)
        self.actuator.set_joint_positions(self.actuator_initial_position)
        return self.get_state()

    def do_step(self, action, model_string):
        self.actuator.set_joint_target_velocities(action)
        self.controller.step()


        if model_string == "base":
            return self.base_article_reward(), self.get_state()
        else:
            return self.get_reward(), self.get_state()


    def base_article_reward(self):

        if(self.done()):
            rw = BASE_REWARD
        else:
            ax, ay, az = self.actuator_tip.get_position()
            tx, ty, tz = self.target.get_position()
            rw = math.e * (-0.25 * np.sqrt((ax-tx)**2+(ay-ty)**2+(az-tz)**2))

        return rw

    def get_reward(self):
        ax, ay, az = self.actuator_tip.get_position()
        tx, ty, tz = self.target.get_position()
        rw = -np.sqrt((ax-tx)**2+(ay-ty)**2+(az-tz)**2)
        return rw

    def shutdown(self):
        return (self.controller.stop()) and (self.controller.shutdown())

    def start_sim(self):
        return self.controller.start()

    def stop_sim(self):
        return self.controller.stop()
