import pyrealsense2 as rs
import numpy as np
import cv2

def returnImage(pipeline) :
    while True :
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
        return color_frame, depth_frame, color_image, depth_image

if __name__ == '__main__':

    WIDTH = 640
    HEIGHT = 480    

    try : 
        # ストリーミング初期化
        config = rs.config()
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

        # ストリーミング開始
        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        # 距離に関するものを取る
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Alignオブジェクト生成
        align_to = rs.stream.color
        align = rs.align(align_to)

        # KCF(トラッキング方法)
        tracker = cv2.TrackerKCF_create()

        # スクショして検出範囲を選択
        while True:
            color_frame, depth_frame, color_image, depth_image = returnImage(pipeline)
            bbox = (0,0,10,10)
            bbox = cv2.selectROI(color_image, False)
            ok = tracker.init(color_image, bbox)
            cv2.destroyAllWindows()
            break

        while True:
            color_frame, depth_frame, color_image, depth_image = returnImage(pipeline)

            # トラッカーをアップデートする
            track, bbox = tracker.update(color_image)

            #depth imageをカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

            # 検出した場所に四角を書く
            if track:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(color_image, p1, p2, (0,255,0), 2, 1)
            else :
                # トラッキングが外れたら警告を表示する
                cv2.putText(color_image, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # 距離を表示する
            dists = []
            for x in range(int(bbox[0]), int(bbox[0] + bbox[2])) :
                for y in range(int(bbox[1]), int(bbox[1] + bbox[3])) :
                    dists.append(depth_frame.get_distance(x, y))
            dist = np.average(dists)

            cv2.putText(color_image, "distance : " + str(dist/depth_scale), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # 加工済の画像を表示する
            color_image_s = cv2.resize(color_image, (WIDTH, HEIGHT))
            depth_colormap_s = cv2.resize(depth_colormap, (WIDTH, HEIGHT))
            images = np.hstack((color_image_s, depth_colormap_s))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # キー入力を1ms待って、k が27（ESC）だったらBreakする
            k = cv2.waitKey(1)
            if k == 27 :
                break

    finally :
        # ストリーミング停止
        pipeline.stop()
