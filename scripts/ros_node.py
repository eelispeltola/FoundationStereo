# ROS 2 node that runs FoundationStereo on a stereo pair (e.g. ZED X).
#
# Subscribes to left/right images (raw or compressed) and the matching camera_info
# topics, optionally rectifies them (rectify:=true for unrectified driver output), and publishes disparity, metric depth and an optional
# colored point cloud. Works the same on a live camera or on `ros2 bag play`.
#
# Usage:
#   source /opt/ros/<distro>/setup.bash
#   python scripts/ros_node.py --ros-args -p scale:=0.5 -p valid_iters:=16
#   ros2 bag play <bag> --rate 0.1      # in another terminal


import os,sys
import logging
import numpy as np
import cv2
import torch
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity
from core.foundation_stereo import FoundationStereo

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, PointField
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import message_filters


ZED_PREFIX = '/zed/zed_node'


class FoundationStereoNode(Node):
  def __init__(self):
    super().__init__('foundation_stereo')
    p = self.declare_parameter
    # ZED wrapper >= 5.0 topic names. For older wrappers use
    # left/image_rect_color, right/image_rect_color, left/camera_info, right/camera_info.
    p('left_topic', f'{ZED_PREFIX}/left/color/rect/image')
    p('right_topic', f'{ZED_PREFIX}/right/color/rect/image')
    p('left_info_topic', f'{ZED_PREFIX}/left/color/rect/camera_info')
    p('right_info_topic', f'{ZED_PREFIX}/right/color/rect/camera_info')
    p('compressed', False)          # subscribe to <topic>/compressed instead of raw images
    p('rectify', False)             # images are unrectified: rectify from both camera_infos (see on_info)
    p('rotation_correction_deg', [0.0, 0.0, 0.0])  # small right-camera rotation fix (rx, ry, rz) on top of camera_info
    p('baseline', 0.0)              # meters; 0 = read from right camera_info P[0,3] (ZED X is ~0.12)
    p('ckpt_dir', f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth')
    p('scale', 1.0)                 # downsize input, must be <=1
    p('max_width', 0)               # downsize further if wider than this (0 = off); 1280 fits 8 GB GPUs
    p('hiera', False)               # hierarchical inference for >1K images
    p('valid_iters', 32)
    p('remove_invisible', True)     # drop pixels not visible in the right image
    p('z_far', 10.0)                # max depth kept in the point cloud
    p('mask_sky', False)            # drop low-texture regions touching the top edge (see sky_mask)
    p('sky_texture_thresh', 4.0)    # mean |Laplacian| below this counts as textureless
    p('publish_cloud', True)
    p('publish_vis', True)
    p('best_effort', False)         # image subscription QoS
    p('sync_slop', 0.01)            # seconds; ZED left/right share stamps, so keep this small
    self.prm = lambda name: self.get_parameter(name).value

    self.bridge = CvBridge()
    self.load_model()

    self.K = None
    self.baseline = self.prm('baseline') or None
    self.rect_maps = None
    self.infos = {}
    self.create_subscription(CameraInfo, self.prm('left_info_topic'), lambda m: self.on_info('left', m), 10)
    self.create_subscription(CameraInfo, self.prm('right_info_topic'), lambda m: self.on_info('right', m), 10)

    msg_type = CompressedImage if self.prm('compressed') else Image
    suffix = '/compressed' if self.prm('compressed') else ''
    # Reliable by default: large images over best-effort stop arriving while the node is busy
    # with inference. Set best_effort:=true only if the publisher is best-effort.
    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT if self.prm("best_effort") else ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    left_sub = message_filters.Subscriber(self, msg_type, self.prm('left_topic')+suffix, qos_profile=qos)
    right_sub = message_filters.Subscriber(self, msg_type, self.prm('right_topic')+suffix, qos_profile=qos)
    self.sync = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=self.prm('sync_slop'))
    self.sync.registerCallback(self.on_stereo)

    self.pub_disp = self.create_publisher(DisparityImage, '~/disparity', 1)
    self.pub_depth = self.create_publisher(Image, '~/depth', 1)
    self.pub_cloud = self.create_publisher(PointCloud2, '~/points', 1) if self.prm('publish_cloud') else None
    self.pub_vis = self.create_publisher(Image, '~/disparity_vis', 1) if self.prm('publish_vis') else None

    # Inference is much slower than the camera rate, so only the newest pair is processed
    # and older ones are dropped instead of building up a backlog (see main()).
    self.latest = None
    self.n_received = 0
    self.n_processed = 0
    self.get_logger().info(f"Waiting for {self.prm('left_topic')+suffix} and {self.prm('right_topic')+suffix}")

  def load_model(self):
    ckpt_dir = self.prm('ckpt_dir')
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
      cfg['vit_size'] = 'vitl'
    cfg['valid_iters'] = self.prm('valid_iters')
    self.model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    self.model.load_state_dict(ckpt['model'])
    self.model.cuda()
    self.model.eval()
    self.get_logger().info(f"Loaded {ckpt_dir} (global_step {ckpt['global_step']})")

  def on_info(self, side, msg):
    self.infos[side] = msg
    if self.prm('rectify'):
      if self.rect_maps is None and len(self.infos)==2:
        self.init_rectification(self.infos['left'], self.infos['right'])
      return
    if side=='left':
      # Rectified images: use the projection matrix rather than the raw K.
      self.K = np.array(msg.p, dtype=np.float32).reshape(3,4)[:,:3].copy()
    elif self.baseline is None:
      P = np.array(msg.p, dtype=np.float64).reshape(3,4)
      if P[0,3] != 0:
        self.baseline = float(-P[0,3]/P[0,0])
        self.get_logger().info(f"Baseline from right camera_info: {self.baseline:.4f} m")

  def init_rectification(self, left, right):
    # Unrectified ZED driver output: each camera_info holds its own K and D, the left has R=I and P=K,
    # and the right has the inter-camera rotation in R and the baseline in P[0,3]. That is not a
    # rectified pair (principal points differ), so compute a proper stereo rectification here.
    # The right R maps right->left, while OpenCV wants left->right, hence the transpose
    # (checked on a ZED X bag: row offset 0.0 px with R^T vs 1.4 px with R).
    K1 = np.array(left.k).reshape(3,3); D1 = np.array(left.d)
    K2 = np.array(right.k).reshape(3,3); D2 = np.array(right.d)
    R = np.array(right.r).reshape(3,3).T
    # Optional correction for calibration error (rotation vector, degrees, applied on the left).
    # ry (yaw) shows up as a constant disparity bias, so calibrate it against lidar or known depths.
    corr = np.radians(np.array(self.prm('rotation_correction_deg'), dtype=np.float64))
    if np.any(corr):
      R = cv2.Rodrigues(corr)[0] @ R
    P = np.array(right.p).reshape(3,4)
    baseline = self.baseline or float(-P[0,3]/P[0,0])
    if baseline <= 0:
      self.get_logger().error("Right camera_info has no baseline in P[0,3]; set the baseline parameter")
      return
    size = (left.width, left.height)
    R1,R2,P1,P2,_,_,_ = cv2.stereoRectify(K1, D1, K2, D2, size, R, np.array([-baseline,0,0]), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    self.rect_maps = [cv2.initUndistortRectifyMap(K, D, Rr, Pr, size, cv2.CV_32FC1) for K,D,Rr,Pr in ((K1,D1,R1,P1),(K2,D2,R2,P2))]
    self.K = P1[:,:3].astype(np.float32)
    self.baseline = baseline
    self.get_logger().info(f"Rectifying {size[0]}x{size[1]} pair, baseline {baseline:.4f} m, f {P1[0,0]:.1f}. "
                           f"Outputs are in the rectified left frame, rotated from {left.header.frame_id} by R1.")

  def to_rgb(self, msg):
    if isinstance(msg, CompressedImage):
      img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
      return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = self.bridge.imgmsg_to_cv2(msg)
    enc = msg.encoding.lower()
    if enc in ('mono8', '8uc1'):
      return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if enc == 'bgra8':
      return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if enc == 'rgba8':
      return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if enc == 'bgr8':
      return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

  def sky_mask(self, rgb, win=15):
    # Stereo has nothing to match in the sky, so the model makes up depth there (often far too close).
    # Mask textureless regions that are connected to the top image border; thin structures such as
    # poles and power lines have texture and survive.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.blur(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3,3), 0), cv2.CV_32F)), (win,win))
    low = (texture < self.prm('sky_texture_thresh')).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    _, labels = cv2.connectedComponents(low)
    top = np.unique(labels[0])
    return np.isin(labels, top[top!=0])

  def on_stereo(self, left, right):
    self.n_received += 1
    self.latest = (left, right)

  def process_latest(self):
    if self.latest is None:
      return
    left, right = self.latest
    self.latest = None
    try:
      self.process(left, right)
    except Exception as e:
      self.get_logger().error(f"Processing failed: {e}")

  @torch.no_grad()
  def infer(self, img0, img1):
    H,W = img0.shape[:2]
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.amp.autocast('cuda', enabled=True):
      if not self.prm('hiera'):
        disp = self.model.forward(img0, img1, iters=self.prm('valid_iters'), test_mode=True)
      else:
        disp = self.model.run_hierachical(img0, img1, iters=self.prm('valid_iters'), test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    return disp.data.cpu().numpy().reshape(H,W)

  def process(self, left, right):
    if self.K is None or self.baseline is None:
      self.get_logger().warn("No camera_info/baseline yet, skipping frame", throttle_duration_sec=2.0)
      return
    scale = self.prm('scale')
    assert scale<=1, "scale must be <=1"
    img0 = self.to_rgb(left)
    img1 = self.to_rgb(right)
    if self.rect_maps is not None:
      img0 = cv2.remap(img0, *self.rect_maps[0], cv2.INTER_LINEAR)
      img1 = cv2.remap(img1, *self.rect_maps[1], cv2.INTER_LINEAR)
    if self.prm('max_width') > 0:
      scale = min(scale, self.prm('max_width')/img0.shape[1])
    if scale != 1:
      img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
      img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    K = self.K.copy()
    K[:2] *= scale

    disp = self.infer(img0, img1)
    if self.prm('remove_invisible'):
      xx = np.arange(disp.shape[1])[None]
      disp[(xx-disp)<0] = np.inf
    if self.prm('mask_sky'):
      disp[self.sky_mask(img0)] = np.inf
    with np.errstate(divide='ignore'):
      depth = (K[0,0]*self.baseline/disp).astype(np.float32)
    depth[~np.isfinite(depth) | (depth<=0)] = np.nan

    header = left.header
    self.publish_disparity(header, disp, K)
    depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
    depth_msg.header = header
    self.pub_depth.publish(depth_msg)
    if self.pub_vis is not None:
      vis = vis_disparity(disp)
      vis_msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(vis), encoding='rgb8')
      vis_msg.header = header
      self.pub_vis.publish(vis_msg)
    if self.pub_cloud is not None:
      self.pub_cloud.publish(self.make_cloud(header, depth, img0, K))

    self.n_processed += 1
    self.get_logger().info(f"Processed {self.n_processed}/{self.n_received} pairs (stamp {header.stamp.sec}.{header.stamp.nanosec:09d})", throttle_duration_sec=5.0)

  def publish_disparity(self, header, disp, K):
    msg = DisparityImage()
    msg.header = header
    d = disp.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    msg.image = self.bridge.cv2_to_imgmsg(d, encoding='32FC1')
    msg.image.header = header
    msg.f = float(K[0,0])
    msg.t = float(self.baseline)
    msg.min_disparity = 0.0
    msg.max_disparity = float(d.max())
    msg.delta_d = 1.0/16
    self.pub_disp.publish(msg)

  def make_cloud(self, header, depth, rgb, K):
    H,W = depth.shape
    vs,us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth
    keep = np.isfinite(z) & (z>0) & (z<=self.prm('z_far'))
    z = z[keep]
    x = (us[keep]-K[0,2])*z/K[0,0]
    y = (vs[keep]-K[1,2])*z/K[1,1]
    c = rgb[keep].astype(np.uint32)
    packed = (c[:,0]<<16) | (c[:,1]<<8) | c[:,2]

    dtype = np.dtype([('x','<f4'), ('y','<f4'), ('z','<f4'), ('rgb','<u4')])
    pts = np.empty(len(z), dtype=dtype)
    pts['x'], pts['y'], pts['z'], pts['rgb'] = x, y, z, packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [
      PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
      PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = dtype.itemsize
    msg.row_step = msg.point_step*msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def main():
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  rclpy.init()
  node = FoundationStereoNode()
  try:
    # Drain pending callbacks so `latest` is the newest pair, then run inference on it.
    while rclpy.ok():
      rclpy.spin_once(node, timeout_sec=0.1)
      for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0)
      node.process_latest()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__=="__main__":
  main()
