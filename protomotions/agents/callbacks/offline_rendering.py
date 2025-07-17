import numpy as np
import imageio
from protomotions.agents.callbacks.base_callback import RL_EvalCallback
from typing import Dict, Any, Optional

# isaaclab imports
from isaaclab.sim import SimulationContext


class IsaacLabOfflineRenderingCallback(RL_EvalCallback):

    def __init__(self, config: Dict[str, Any], training_loop):
        super().__init__(config, training_loop)
        self.env = training_loop.env
        self.sim: SimulationContext = self.env.simulator._sim

        # -- Configuration from dict --
        self.video_width = self.config.get("video_width", 960)
        self.video_height = self.config.get("video_height", 1280)
        self.camera_offset = np.array(self.config.get("camera_offset", [0.8, -0.8, 0.5]))
        self.video_filename = self.config.get("video_filename", "eval.mp4")
        self.camera_prim_path = self.config.get("camera_prim_path", "/OmniverseKit_Persp")
        self.robot_articulation = self.env.simulator._scene.articulations["robot"]

        # -- Runtime variables --
        self.video_writer = None
        self.rgb_annotator = None
        self.render_product = None

    def _setup_camera_and_writer(self) -> None:
        """Initializes the replicator annotator and the video writer."""
        import omni.replicator.core as rep
        self.render_product = rep.create.render_product(
            self.camera_prim_path, (self.video_width, self.video_height)
        )
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb_annotator.attach([self.render_product])

        fps = int(1.0 / self.sim.get_physics_dt())
        self.video_writer = imageio.get_writer(self.video_filename, fps=fps)

    def _update_camera_and_capture_frame(self) -> None:
        """Updates camera position to follow the robot and captures a single frame."""
        base_pos = self.robot_articulation.data.root_pos_w[0].cpu().numpy()
        eye_pos = base_pos + self.camera_offset
        self.sim.set_camera_view(eye=eye_pos.tolist(), target=base_pos.tolist())

        # Step the replicator pipeline to capture the rendered data
        self.sim.render()
        rgb_data = self.rgb_annotator.get_data()

        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self.video_height, self.video_width, 4)
        self.video_writer.append_data(frame[..., :3])

    def _close_writer_and_cleanup(self) -> None:
        """Closes the video writer and cleans up replicator resources."""
        self.video_writer.close()
        self.rgb_annotator.detach(self.render_product)

    def on_pre_evaluate_policy(self):
        """Called before policy evaluation begins. Sets up rendering."""
        self._setup_camera_and_writer()

    def on_post_eval_env_step(self, actor_state: Any):
        """Called after each environment step during evaluation. Captures a frame."""
        self._update_camera_and_capture_frame()
        return actor_state

    def on_post_evaluate_policy(self):
        """Called after policy evaluation ends. Finalizes the video."""
        self._close_writer_and_cleanup()