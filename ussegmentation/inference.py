"""Inference implementations."""
import logging
import time

import cv2

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

    def __enter__(self):
        if self.file_name == "":
            self.stream = cv2.VideoCapture(0)
        else:
            self.stream = cv2.VideoCapture(self.file_name)
        return self.stream

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
    """Base class for inference."""

    input_types = ["video", "image"]

    def __init__(self):
        self.log = logging.getLogger(__name__)

    def run(self, input_type, input_file, output_file, show):
        """Performs inference on specified data."""
        if input_type == "video":
            self._process_video(input_file, output_file, show)
        if input_type == "image":
            self._process_image(input_file, output_file, show)

    def _process_video(self, input_file, output_file, show):
        previewer = Previewer(show)
        with SourceVideo(input_file) as source:
            with DestinationVideo(output_file, source) as destination:
                next_frame_exist, input_frame = source.read()
                while (
                    source.isOpened()
                    and next_frame_exist
                    and previewer.is_work()
                ):
                    output_frame = self.inference(input_frame)
                    destination.write(output_frame)
                    previewer.show(output_frame)
                    next_frame_exist, input_frame = source.read()

    def _process_image(self, input_file, output_file, show):
        previewer = Previewer(show)
        input_frame = cv2.imread(input_file)
        output_frame = self.inference(input_frame)
        if output_file != "":
            cv2.imwrite(output_file, output_frame)
        previewer.show(output_frame)
        cv2.waitKey(0)

    @execution_time
    def inference(self, input):
        """Produce an inference on input data and return result."""
        raise NotImplementedError


class EmptyInference(Inference):
    """Simple empty inference to check."""

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("Run an empty inference")

    @execution_time
    def inference(self, input):
        """Simple HSV conversation to check inference."""
        return cv2.cvtColor(input, cv2.COLOR_BGR2HSV)


class InferenceCreator:
    """Class to create selected inference object."""

    inferences = {"empty": EmptyInference}
    input_types = Inference.input_types

    def __init__(self, inference_type):
        self.type = inference_type

    def create_inference(self):
        """Factory methon for inference."""
        return self.inferences[self.type]()