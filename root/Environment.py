from pyrep import PyRep
from os.path import dirname, join, abspath
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np, cv2 as cv
from PIL import Image as I


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


    def reset_scene(self):
        new_target_pos = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        self.target.set_position(new_target_pos)
        self.actuator.set_joint_positions(self.actuator_initial_position)
        return self.get_state()

    def do_step(self, action):
        self.actuator.set_joint_target_velocities(action)
        self.controller.step()
        return self.get_reward(), self.get_state()

    def get_reward(self):
        ax, ay, az = self.actuator_tip.get_position()
        tx, ty, tz = self.target.get_position()
        return -np.sqrt((ax-tx)**2+(ay-ty)**2+(az-tz)**2)

    def shutdown(self):
        return (self.controller.stop()) and (self.controller.shutdown())

    def start_sim(self):
        return self.controller.start()

    def stop_sim(self):
        return self.controller.stop()
