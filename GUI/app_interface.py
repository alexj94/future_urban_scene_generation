from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QRect
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget

from trajectory_inference import traj_test
from utils.bounding_box import BoundingBox


class main_GUI(QWidget):
    def __init__(self, title: str, cap, video_dir: Path, trajectories, scale,
                 inv_homo_matrix, args, device, config, maskrcnn_mode, edge_model, inpaint_model,
                 model_cad, model_kp, model_icn, model_VUnet, cads_ply, kpoints_dicts,
                 inpaint_flag):
        """
        GUI for performing inference in the two possible modes
        :param title: Window name
        :param cap: OpenCV video capture
        :param video_dir: Video directory
        :param trajectories: Vehicles trajectories for current video
        :param scale: Video scale resize value (1920x1080 to 1280x720)
        :param inv_homo_matrix: Inverse homography matrix (pixel to GPS coordinates)
        :param config: EdgeConnect configuration
        :param edge_model: EdgeConnect edge model
        :param inpaint_model: EdgeConnect inpaint model
        :param model_cad: VGG19-based cad classifier model
        :param model_kp: Stacked Hourglass model for keypoint localization
        """
        super().__init__()

        self.title = title
        self.cap = cap
        self.video_dir = video_dir
        self.trajectories = trajectories
        self.scale = scale
        self.inv_homo_matrix = inv_homo_matrix
        self.args = args
        self.device = device
        self.config = config
        self.maskrcnn_model = maskrcnn_mode
        self.edge_model = edge_model
        self.inpaint_model = inpaint_model
        self.model_cad = model_cad
        self.model_kp = model_kp
        self.model_icn = model_icn
        self.model_VUnet = model_VUnet
        self.cads_ply = cads_ply
        self.kpoints_dicts = kpoints_dicts
        self.inpaint_flag = inpaint_flag

        self.curr_frame = 0
        self.frame = None
        self.orig_frame = None
        self.img_scale = None
        self.shape = None
        self.pic = None
        self.selected_id = []
        self.selected_bbox = []

        self.curr_frame_label = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        # grid layout
        self.create_grid_layout()

        self.showMaximized()

    def create_grid_layout(self):
        window_grid_layout = QGridLayout(self)

        first_v_layout = QVBoxLayout(self)
        first_v_layout.setSpacing(30)
        second_v_layout = QVBoxLayout(self)
        second_v_layout.setSpacing(20)

        info_v_layout = QVBoxLayout(self)
        info_v_layout.setSpacing(20)
        button_v_layout = QVBoxLayout(self)
        button_v_layout.setSpacing(30)
        warning_v_layout = QVBoxLayout(self)
        warning_v_layout.setSpacing(5)
        button_h_layout = QHBoxLayout(self)
        button_h_layout.setSpacing(20)

        # COMMANDS INFO
        first_line_instr = QLabel("Commands working in this window:")
        first_line_instr.setFont(QtGui.QFont('Sans Serif', 16))
        second_line_instr = QLabel("1. Press 'RIGHT ARROW' to go to next frame")
        second_line_instr.setFont(QtGui.QFont('Sans Serif', 16))
        third_line_instr = QLabel("2. Press 'LEFT ARROW' to go to previous frame")
        third_line_instr.setFont(QtGui.QFont('Sans Serif', 16))
        fourth_line_instr = QLabel("3. 'SINGLE MOUSE CLICK' on bounding box"
                                   "\n    to visualize future car trajectory")
        fourth_line_instr.setFont(QtGui.QFont('Sans Serif', 16))
        fifth_line_instr = QLabel("4. Press 'BACKSPACE' to cancel drawn trajectories \n"
                                  "    and reset parameters to perform a new test")
        fifth_line_instr.setFont(QtGui.QFont('Sans Serif', 16))

        info_v_layout.addWidget(first_line_instr, alignment=Qt.AlignLeft)
        info_v_layout.addWidget(second_line_instr, alignment=Qt.AlignLeft)
        info_v_layout.addWidget(third_line_instr, alignment=Qt.AlignLeft)
        info_v_layout.addWidget(fourth_line_instr, alignment=Qt.AlignLeft)
        info_v_layout.addWidget(fifth_line_instr, alignment=Qt.AlignLeft)

        # EXECUTION BUTTONS
        button_info_label = QLabel("'DOUBLE CLICK' to select the bounding box of the\n"
                                   "vehicles on which you want to test the method.\n"
                                   "After that, click on 'RUN' button.")
        button_info_label.setFont(QtGui.QFont('Sans Serif', 16, QtGui.QFont.Cursive))
        inference_button = PushButton('RUN')
        inference_button.setFocusPolicy(Qt.NoFocus)
        inference_button.setFixedWidth(200)
        inference_button.setFixedHeight(40)
        inference_button.setFont(QtGui.QFont('Sans Serif', 12, QtGui.QFont.Bold))
        inference_button.setStyleSheet('color: red')
        inference_button.clicked.connect(self.perform_test)
        results_label = QLabel('Predictions will be saved in the directory \n'
                               '"/results" of the project!')
        results_label.setFont(QtGui.QFont('Sans Serif', 16, QtGui.QFont.Bold))
        warning_label = QLabel('WARNING!')
        warning_label.setFont(QtGui.QFont('Sans Serif', 16, QtGui.QFont.Bold))
        warning_label.setStyleSheet('color: red')
        note_label = QLabel('After every prediction is mandatory to reset the\n'
                            'parameters by clicking "BACKSPACE".')
        note_label.setFont(QtGui.QFont('Sans Serif', 16, QtGui.QFont.Cursive))

        button_v_layout.addWidget(button_info_label, alignment=Qt.AlignLeft)
        button_v_layout.addWidget(inference_button, alignment=Qt.AlignCenter)
        button_v_layout.addWidget(results_label, alignment=Qt.AlignLeft)
        warning_v_layout.addWidget(warning_label, alignment=Qt.AlignLeft)
        warning_v_layout.addWidget(note_label, alignment=Qt.AlignLeft)
        button_v_layout.addLayout(warning_v_layout)

        first_v_layout.addLayout(info_v_layout)
        first_v_layout.addLayout(button_v_layout)

        # VIDEO LAYOUT
        video_label = QLabel(f'Cityflow video "{self.video_dir._cparts[-1]}" '
                             f'from scene "{self.video_dir._cparts[-2]}"')
        video_label.setFont(QtGui.QFont('Sans Serif', 20, QtGui.QFont.ExtraBold))
        video_label.setGeometry(QRect(0, 0, 1281, 20))
        self.curr_frame_label = QLabel(f'Current frame: {self.curr_frame}')
        self.curr_frame_label.setFont(QtGui.QFont('Sans Serif', 16, QtGui.QFont.Bold))
        self.curr_frame_label.setStyleSheet('color: red')

        second_v_layout.addWidget(video_label, alignment=Qt.AlignCenter)
        second_v_layout.addWidget(self.curr_frame_label, alignment=Qt.AlignCenter)
        second_v_layout.addWidget(self.visualize_image(), alignment=Qt.AlignCenter)

        window_grid_layout.addLayout(first_v_layout, 0, 0, alignment=Qt.AlignCenter)
        window_grid_layout.addLayout(second_v_layout, 0, 1, alignment=Qt.AlignCenter)

    def visualize_image(self):
        """
        Initialization of first frame of the video
        """
        # Read frame and initialize frame variables
        self.curr_frame += 1
        ret, frame = self.cap.read()
        while self.curr_frame != self.trajectories[0, 0]:
            self.curr_frame += 1
            ret, frame = self.cap.read()
        orig_h, orig_w, _ = frame.shape
        frame = cv2.resize(frame, (1280, 720))
        h, w, _ = frame.shape

        self.frame = frame
        self.orig_frame = frame.copy()
        self.shape = [w, h]
        self.img_scale = w / orig_w

        # Trajectories of current frame
        car_trajs = self.trajectories[self.trajectories[:, 0] == self.curr_frame, :]

        # Draw all bounging boxes in current frame that have tracking informations
        for det in car_trajs:
            BoundingBox(*det[2:6] * self.img_scale, bounds=(0, w - 1, 0, h - 1),
                        scale=self.scale).draw(self.frame, color=(0, 255, 0))

        # Attach image to window
        self.pic = QLabel(self)
        nparray_to_pixmap = self.convert_numpy_to_pixmap(frame)
        self.pic.setPixmap(nparray_to_pixmap)
        self.pic.mousePressEvent = self.get_pos

        self.curr_frame_label.setText(f'Current frame: {self.curr_frame}')

        return self.pic

    @staticmethod
    def convert_numpy_to_pixmap(np_img):
        """
        Convert numpy array image to PixMap
        """
        height, width, channel = np_img.shape
        bytes_per_line = 3 * width
        return QPixmap(QImage(np_img.data, width, height,
                              bytes_per_line, QImage.Format_RGB888).rgbSwapped())

    # EVENT HANDLERS
    def perform_test(self):
        """
        Handle click event on test button
        """
        if self.selected_bbox is not None:
            future_trajs = []

            for curr_id in self.selected_id:
                future_locs_idxs = np.bitwise_and(self.trajectories[:, 1] == curr_id,
                                                  self.trajectories[:, 0] >= self.curr_frame)
                curr_future_trajs = self.trajectories[future_locs_idxs]
                curr_trajs = []
                for i in range(0, 11, 2):
                    if i > curr_future_trajs.shape[0] - 1:
                        break
                    curr_trajs.append(curr_future_trajs[i])
                future_trajs.append(np.asarray(curr_trajs))

            # Perform trajectory mode inference
            self.cap = traj_test(self.args, self.cap, self.curr_frame, self.orig_frame.copy(),
                                 self.selected_bbox, future_trajs, self.inv_homo_matrix, self.scale,
                                 self.img_scale, self.device, self.config, self.maskrcnn_model, self.edge_model,
                                 self.inpaint_model, self.model_cad, self.model_kp, self.model_icn,
                                 self.model_VUnet, self.cads_ply, self.kpoints_dicts,
                                 self.inpaint_flag)

    def get_pos(self, event):
        """
        Handle click events on bounding boxes
        """
        x = event.pos().x()
        y = event.pos().y()

        for curr_traj in self.trajectories[self.trajectories[:, 0] == self.curr_frame, :]:
            cur_bbox = BoundingBox(*curr_traj[2:6] * self.img_scale,
                                   bounds=(0, self.shape[0] - 1, 0, self.shape[1] - 1),
                                   scale=self.scale)

            if cur_bbox.contains((x, y)):
                vehicle_id = curr_traj[1]

                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.LeftButton:
                        # Draw future trajectory for current vehicle
                        future_locs_idxs = np.bitwise_and(self.trajectories[:, 1] == vehicle_id,
                                                          self.trajectories[:, 0] >= self.curr_frame)
                        future_locs = self.trajectories[future_locs_idxs]

                        future_midpoints = [BoundingBox(*fl[2:6] * self.img_scale,
                                                        bounds=(0, self.shape[0] - 1,
                                                                0, self.shape[1] - 1),
                                                        scale=self.scale).mid_bottom
                                            for fl in future_locs]
                        cv2.polylines(self.frame, np.asarray(future_midpoints)[np.newaxis],
                                      0, (255, 0, 0), 2)
                        self.pic.setPixmap(self.convert_numpy_to_pixmap(self.frame))

                elif event.type() == QEvent.MouseButtonDblClick:
                    cur_bbox.draw(self.frame, (0, 0, 255))
                    self.selected_id.append(curr_traj[1])
                    self.selected_bbox.append(cur_bbox.xyxy)
                    self.pic.setPixmap(self.convert_numpy_to_pixmap(self.frame))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """
        Handle keyboard pressed buttons
        """
        if event.key() == Qt.Key_Backspace:
            self.selected_id = []
            self.selected_bbox = []

            self.frame = self.orig_frame.copy()

            for det in self.trajectories[self.trajectories[:, 0] == self.curr_frame, :]:
                BoundingBox(*det[2:6] * self.img_scale,
                            bounds=(0, self.shape[0] - 1, 0, self.shape[1] - 1),
                            scale=self.scale).draw(self.frame, color=(0, 255, 0))

            self.pic.setPixmap(self.convert_numpy_to_pixmap(self.frame))

        elif event.key() == Qt.Key_Right:
            self.curr_frame += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame - 1)
            ret, frame = self.cap.read()
            if not ret:
                return
            frame = cv2.resize(frame, (1280, 720))

            self.frame = frame
            self.orig_frame = frame.copy()

            # Trajectories of current frame
            car_trajs = self.trajectories[self.trajectories[:, 0] == self.curr_frame, :]

            # Draw all bounging boxes in current frame that have tracking informations
            for det in car_trajs:
                BoundingBox(*det[2:6] * self.img_scale,
                            bounds=(0, self.shape[0] - 1, 0, self.shape[1] - 1),
                            scale=self.scale).draw(self.frame, color=(0, 255, 0))

            self.curr_frame_label.setText(f'Current frame: {self.curr_frame}')
            self.pic.setPixmap(self.convert_numpy_to_pixmap(self.frame))

        elif event.key() == Qt.Key_Left:
            if self.curr_frame > 1:
                self.curr_frame -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame - 1)
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (1280, 720))

                self.frame = frame
                self.orig_frame = frame.copy()

                # Trajectories of current frame
                car_trajs = self.trajectories[self.trajectories[:, 0] == self.curr_frame, :]

                # Draw all bounging boxes in current frame that have tracking informations
                for det in car_trajs:
                    BoundingBox(*det[2:6] * self.img_scale,
                                bounds=(0, self.shape[0] - 1, 0, self.shape[1] - 1),
                                scale=self.scale).draw(self.frame, color=(0, 255, 0))

                self.curr_frame_label.setText(f'Current frame: {self.curr_frame}')
                self.pic.setPixmap(self.convert_numpy_to_pixmap(self.frame))


class PushButton(QPushButton):
    def __init__(self, *args):
        QPushButton.__init__(self, *args)

    def event(self, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            return True

        return QPushButton.event(self, event)


class CheckBox(QCheckBox):
    def __init__(self, *args):
        QCheckBox.__init__(self, *args)

    def event(self, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            return True

        return QCheckBox.event(self, event)
