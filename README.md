# Future urbane scene generation through vehicle synthesis

[RESULTS IMAGES]

## Abstract

In this work we propose a deep learning pipeline to predict 
the visual future appearance of an urban scene. Despite 
recent advances, generating the entire scene in an 
end-to-end fashion is still far from being achieved. 
Instead, here we follow a two stage approach, where 
interpretable information are included in the loop and 
each actor is modelled independently. We leverage a 
per-object *novel view synthesis* paradigm; i.e. 
generating a synthetic representation of an object 
undergoing a geometrical roto-translation in the 3D space. 
Our model can be easily conditioned with constraints (e.g. 
input trajectories) provided by state-of-the-art tracking 
methods or by the user. We visually and quantitatively 
show the superiority of this approach over traditional 
end-to-end scene-generation methods on CityFlow, a real 
world dataset.

<p align="center">
  <img src="imgs/model.svg"/ alt="Multi stage pipeline">
</p>

---

## Code

### Install

Run the following command to install all requirements in a 
new virtual environment:

```bash
pip install -r requirements.txt
```

In order to make working the MaskRCNN, go to directory 
`detectron2` and then execute this command using the 
virtual environment that you just created:

```bash
python -m pip install -e .
```

Code was tested with Python 3.6 on an Ubuntu Linux based 
system.

### How to run test

To run the demo of our project, please firstly download all 
the required data at this **link** and save them in a
`<data_dir>` of your choice. We tested our pipeline on 
**CityFlow** dataset that already have annotated bounding 
boxes and trajectories of vehicles.

The test script is `gui_test.py` that expects some 
arguments as mandatory: video, 3D keypoints and checkpoints 
directories.

```bash
python gui_test.py <data_dir>/<video_dir> <data_dir>/3Dkpoints <data_dir>/checkpoints --det_mode ssd512|yolo3|mask_rcnn --track_mode tc|deepsort|moana --bbox_scale 1.15 --device cpu|cuda
```

Add the parameter `--inpaint` to use the inpainting on the 
vehicle instead of the background suppression.

### Description and GUI usage

If everything went well, you should see the main GUI in 
which you can choose whichever vehicle you want that 
was detected in the video frame or change the video frame.

[GUI image]

The commands working on this window are:
1) `RIGHT ARROW` = go to next frame
2) `LEFT ARROW` = go to previous frame
3) `SINGLE MOUSE LEFT BUTTON CLICK` = visualize car 
trajectory
4) `BACKSPACE` = delete the drawn trajectories
5) `DOUBLE MOUSE LEFT BUTTON CLICK` = select one of the 
vehicles bounding boxes

Once you selected some vehicles of your chioce by 
double-clicking in their bounding boxes, you can push the 
`RUN` button to start the inference. The resulting frames 
will be saved in `./results` directory.
