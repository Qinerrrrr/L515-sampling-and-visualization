# from cv2 import pointPolygonTest
import pyrealsense2 as rs
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import time
import os


class Realsense(object):
    def __init__(self):
        self.init_device()
        self.init_data()

    def init_device(self):
        self.pipeline = rs.pipeline()

        config = rs.config()
        # from the camera l515
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.infrared, 320, 240, rs.format.y8, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30) 
#TODO:
        align_to = rs.stream.color       
        self.align = rs.align(align_to)
        self.pc = rs.pointcloud()

        # start
        self.profile = self.pipeline.start(config)

        ######################################################
        depth_profile = rs.video_stream_profile(self.pipeline.get_active_profile().get_stream(rs.stream.depth))
        self.depth_intrinsics = depth_profile.get_intrinsics()

        ######################################################
        self.sensor = self.profile.get_device().first_depth_sensor()

        #### Short Range ####
        self.sensor.set_option(rs.option.digital_gain, 2)
        self.sensor.set_option(rs.option.laser_power, 87.0)
        self.sensor.set_option(rs.option.confidence_threshold, 1.0)
        self.sensor.set_option(rs.option.min_distance, 95.0)
        self.sensor.set_option(rs.option.receiver_gain, 18.0)
        self.sensor.set_option(rs.option.post_processing_sharpening, 1.0)
        self.sensor.set_option(rs.option.pre_processing_sharpening, 0.0)
        self.sensor.set_option(rs.option.noise_filtering, 4.0)

        #### Low Ambient Light ####
        # self.sensor.set_option(rs.option.digital_gain, 2)
        # self.sensor.set_option(rs.option.laser_power, 100.0)
        # self.sensor.set_option(rs.option.confidence_threshold, 1.0)
        # self.sensor.set_option(rs.option.min_distance, 190.0)
        # self.sensor.set_option(rs.option.receiver_gain, 18.0)
        # self.sensor.set_option(rs.option.post_processing_sharpening, 1.0)
        # self.sensor.set_option(rs.option.pre_processing_sharpening, 0.0)
        # self.sensor.set_option(rs.option.noise_filtering, 4.0)

        ######################################################

        self.scale = self.sensor.get_depth_scale()
        # print(self.sensor.get_supported_options())

    def init_data(self):
        self.frame_align = None
        self.points = None

    def get_data(self):
        try:
            frames = self.pipeline.wait_for_frames()  # wait for two parallel frames
            frame_align = self.align.process(frames)
            self.frame_align = frame_align

            depth_frame = frame_align.get_depth_frame()
            color_frame = frame_align.get_color_frame()

            # self.depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

            points = self.pc.calculate(depth_frame)
            self.points = points

            pts_n = points.get_vertices()
            npts = np.asanyarray(pts_n).view(np.float32).reshape(-1, 3)
            # color_img = np.asanyarray(color_frame.get_data())

            npts = npts * 1000
            
            # depth =  np.asanyarray(depth_frame.get_data(), dtype=np.float64)

            return depth_frame, npts, color_frame, points
        
        except Exception as e:
            print(Exception, e)
            self.frame_align = None
            self.points = None
            return None, None, None

    def get_depth_intrinsics(self):
        width_height = (self.depth_intrinsics.width, self.depth_intrinsics.height)
        principal_point = (self.depth_intrinsics.ppx, self.depth_intrinsics.ppy)
        focal_length = (self.depth_intrinsics.fx, self.depth_intrinsics.fy)
        model = self.depth_intrinsics.model.__str__()
        coeffs = self.depth_intrinsics.coeffs

        # print(self.depth_intrinsics)

        return width_height, principal_point, focal_length, model, coeffs



if __name__ == '__main__':
    
    # init camera
    camera_obj = Realsense()
    width_height, principal_point, focal_length, model, coeffs = camera_obj.get_depth_intrinsics()
    # print('wh:{},pp:{},{},{},{}'.format(width_height, principal_point, focal_length, model, coeffs))
    
    # status
    pause = False
    auto_saving = True
    start_save_nose = False
    start_save_throat = False
    
    # save every ? frame
    save_freq = 5
    
    # create dataset folder
    
    if not os.path.exists('./throat_dataset'):
        os.mkdir('throat_dataset')
    if not os.path.exists('./throat_dataset/depth'):
        os.mkdir('./throat_dataset/depth')
    if not os.path.exists('./throat_dataset/color'):
        os.mkdir('./throat_dataset/color')
    if not os.path.exists('./throat_dataset/points'):
        os.mkdir('./throat_dataset/points')    
    
    if not os.path.exists('./nose_dataset'):
        os.mkdir('nose_dataset')
    if not os.path.exists('./nose_dataset/depth'):
        os.mkdir('./nose_dataset/depth')
    if not os.path.exists('./nose_dataset/color'):
        os.mkdir('./nose_dataset/color')
    if not os.path.exists('./nose_dataset/points'):
        os.mkdir('./nose_dataset/points')
             
    i = 0
    n_timer = 0
    t_timer = 0
    
    while True:
        
        # if not pause:

        depth, npts, color, points = camera_obj.get_data()
        # print(color.shape)
        
        depth_img = np.asarray(depth.get_data(), dtype=np.uint16)
        color_img = np.asarray(color.get_data(), dtype=np.uint8)
        color_img = color_img[:,:,::-1]
        
        

        
        
        ###########################################################
        # point = rs.rs2_deproject_pixel_to_point(
        #     camera_obj.depth_intrinsics,
        #     [320 ,240], 
        #     round(depth.get_distance(320, 240)*100, 2)
        # )
        # print(point)
        # print(round(depth.get_distance(320, 240)*100, 2))
        ###########################################################
        
        if start_save_nose:
            t = time.time()
            if i % save_freq == 0:
                cv2.imwrite('./nose_dataset/depth/{}.png'.format(t), depth_img)
                cv2.imwrite('./nose_dataset/color/{}.png'.format(t), color_img)
                np.save('./nose_dataset/points/{}.npy'.format(t), npts, allow_pickle=True)
                # points.export_to_ply('./dataset/pointcloud/{}.ply'.format(t), color)
            i += 1
            n_timer += 1
            
        if n_timer % 15 == 0:
            start_save_nose == False
        
    
        if start_save_throat:
            t = time.time()
            if i % save_freq == 0:
                cv2.imwrite('./throat_dataset/depth/{}.png'.format(t), depth_img)
                cv2.imwrite('./throat_dataset/color/{}.png'.format(t), color_img)
                np.save('./throat_dataset/points/{}.npy'.format(t), npts, allow_pickle=True)
                # points.export_to_ply('./dataset/pointcloud/{}.ply'.format(t), color)
            i += 1
        
        if t_timer % 15 == 0:
            start_save_throat == False
            
    
        key = cv2.waitKey(1)
        
        if key == ord("n"):
            start_save_nose = True
            # auto_saving = True
        
        if key == ord('t'):
            start_save_throat = True
        
        
        #空格
        # if key == 32:
        #     i = 0
        #     start_save_nose = False
        #     start_save_throat = False
        
        # if key == ord('m'):
        #     # manully save
        #     # auto_saving = False
        #     start_save = False
        #     t = time.time()
            
        #     cv2.imwrite('./dataset/depth/{}.png'.format(t), depth_img)
        #     cv2.imwrite('./dataset/color/{}.png'.format(t), color_img)
        #     np.save('./dataset/points/{}.npy'.format(t), npts, allow_pickle=True)
        #     # points.export_to_ply('./dataset/pointcloud/{}.ply'.format(t), color)
        
        
            
        # if key == ord('p'):
        #     pause = not pause
            
        #Esc    
        if key == 27:
            start_save = False
            break
        
        cv2.imshow('depth', depth_img)
        cv2.putText(color_img, 'nose_data:{}  throat_data:{}  thanks'.format(n_timer,t_timer), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2 )
        cv2.imshow("RGB", color_img)        
    cv2.destroyAllWindows()

