"""Inference implementations."""
import logging
import time

import torch
import cv2

from pathlib import Path

from ussegmentation.datasets.utils import remap_classes_to_colors

measurements = []


def execution_time(func):
    """Decorator to measure an inference time."""

    def time_it(*args, **kwargs):
        log = logging.getLogger(__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        measurements.append(stop_time - start_time)
        log.info(
            "%9d. time %.3f ms, %10.3f fps, avg %.3f ms, %10.3f avg fps",
            len(measurements),
            1000 * (stop_time - start_time),
            1 / (stop_time - start_time)
            if (stop_time - start_time)
            else 1000,
            1000 * (sum(measurements) / len(measurements)),
            len(measurements) / sum(measurements),
        )
        return result

    return time_it


class SourceVideo:
    """Contex manager for source video."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.log = logging.getLogger(__name__)

    def __enter__(self):
        if self.file_name == "":
            self.stream = cv2.VideoCapture(0)
        else:
            if not Path(self.file_name).exists():
                self.log.error("File %s is not exist", self.file_name)
            self.stream = cv2.VideoCapture(self.file_name)
        return self

    def read(self):
        next_frame_exist, input_frame = self.stream.read()
        return next_frame_exist, input_frame

    def is_opened(self):
        return self.stream.isOpened()

    def get(self, item):
        return self.stream.get(item)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.release()
        cv2.destroyAllWindows()


class DestinationVideo:
    """Contex manager for destination video."""

    def __init__(self, file_name, source_video):
        self.file_name = file_name
        self.source_video = source_video
        self.destination = None

    def __enter__(self):
        if self.file_name != "":
            self.destination = cv2.VideoWriter(
                self.file_name,
                int(self.source_video.get(cv2.CAP_PROP_FOURCC)),
                self.source_video.get(cv2.CAP_PROP_FPS),
                (
                    int(self.source_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.source_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
        return self

    def write(self, frame):
        """Save a frame to the output video."""
        if self.destination is not None:
            self.destination.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.destination is not None:
            self.destination.release()


class Previewer:
    """OpenCV previewer for images and video."""

    def __init__(self, show):
        self.work = True
        self.show_window = show
        self.log = logging.getLogger(__name__)
        self.log.info("Press 'q' to exit")

    def is_work(self):
        """Exit feature ov viewer."""
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.work = False

        return self.work

    def show(self, image):
        """Show an inference."""
        if self.show_window:
            cv2.imshow("Inference", image)


class Inference:
    """Class for inference."""

    input_types = ["video", "image"]

    def __init__(self, model, model_file=""):
        self.log = logging.getLogger(__name__)
        self.model = model
        self.load_cpu_model(model_file)
        self.log.info("Run an inference")

    def run(self, input_type, input_file, output_file, show):
        """Performs inference on specified data."""
        if input_type == "video":
            self._process_video(input_file, output_file, show)
        if input_type == "image":
            self._process_image(input_file, output_file, show)

    def _blend(self, input, output):
        alpha = 0.5
        beta = 1.0 - alpha
        return cv2.addWeighted(input, alpha, output, beta, 0.0)

    def _process_video(self, input_file, output_file, show):
        previewer = Previewer(show)
        with SourceVideo(input_file) as source:
            with DestinationVideo(output_file, source) as destination:
                next_frame_exist, input_frame = source.read()
                while (
                    source.is_opened()
                    and next_frame_exist
                    and previewer.is_work()
                ):
                    output_frame = self.inference(input_frame)
                    output_frame = self._blend(input_frame, output_frame)
                    destination.write(output_frame)
                    previewer.show(output_frame)
                    next_frame_exist, input_frame = source.read()

    def _process_image(self, input_file, output_file, show):
        previewer = Previewer(show)
        input_frame = cv2.imread(input_file)
        output_frame = self.inference(input_frame)
        output_frame = self._blend(input_frame, output_frame)
        if output_file != "":
            cv2.imwrite(output_file, output_frame)
        previewer.show(output_frame)
        cv2.waitKey(0)

    def to_torch(input_numpy):
        """Convert numpy array to expected torch"""
        input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)
        input_torch = torch.from_numpy(input_numpy).float()
        input_torch = input_torch.unsqueeze(0)
        input_torch = torch.transpose(input_torch, 2, 3)
        input_torch = torch.transpose(input_torch, 1, 2)
        return input_torch

    def to_numpy(output_torch):
        output_torch = torch.transpose(output_torch, 0, 1)
        output_torch = torch.transpose(output_torch, 1, 2)

        output_numpy = output_torch.detach().numpy()
        output_numpy = remap_classes_to_colors(output_numpy)
        output_numpy = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR)
        return output_numpy

    def load_cpu_model(self, state_file=""):
        """Load saved on empty model to CPU device."""
        device = torch.device("cpu")
        if state_file != "":
            self.model.load_state_dict(
                torch.load(state_file, map_location=device)
            )
        self.model.eval()

    @execution_time
    def inference(self, input_value):
        """Run the simple network."""
        output = self.model(Inference.to_torch(input_value))
        _, class_id = torch.max(output, 1)
        return Inference.to_numpy(class_id)
