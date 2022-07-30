import pyrealsense2 as rs
import numpy as np
import cv2

def click_pos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = param[0]
        depth = param[1]
        cv2.circle(color,center=(x,y),radius=5,color=255,thickness=-1)
        pos_str='(x,y)=('+str(x)+','+str(y)+')'
        cv2.putText(color,pos_str,(x+10, y+10),cv2.FONT_HERSHEY_PLAIN,2,255,2,cv2.LINE_AA)
        images = np.hstack((color, depth))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

WIDTH = 640
HEIGHT = 480

# ストリーミング初期化
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        #imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        #depth imageをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        #画像表示
        color_image_s = cv2.resize(color_image, (640, 360))
        depth_colormap_s = cv2.resize(depth_colormap, (640, 360))
        param = [color_image_s, depth_colormap_s]
        images = np.hstack((color_image_s, depth_colormap_s))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.setMouseCallback('RealSense', click_pos, param)

        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            break

finally:

    #ストリーミング停止
    pipeline.stop()