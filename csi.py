import cv2
import numpy as np
import time
import math
import libcamera
from picamera2 import Picamera2

# 物理尺寸参数
A4_WIDTH = 17.3  # A4纸实际宽度(单位:厘米)
A4_HEIGHT = 25.8  # A4纸实际高度(单位:厘米)
F = 1883  # 相机测量焦距

# 图像处理参数
GAUSSIAN_KERNEL = (5, 5)  # 高斯模糊核大小
BILATERAL_FILTER_PARAMS = (9, 75, 75)  # 双边滤波参数
ADAPTIVE_THRESH_PARAMS = (11, 2)  # 自适应阈值参数(块大小, 常数)
CANNY_THRESHOLDS = (50, 150)  # Canny边缘检测阈值

# 轮廓检测参数
MIN_CONTOUR_AREA = 20000  # 最小轮廓面积(过滤噪声)
MIN_ROI_CONTOUR_AREA = 1000  # ROI区域最小轮廓面积
APPROX_POLY_EPSILON = 0.02  # 多边形逼近系数
RECTANGLE_ERROR_TOLERANCE = 0.15  # 矩形判断容差
CIRCULARITY_THRESHOLD = 0.7  # 圆形判断阈值

# 显示参数
DISPLAY_EDGE_LIMIT = 5  # 最大显示边数量
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
FONT_SIZE_SMALL = 0.5  # 小字体大小
FONT_SIZE_MEDIUM = 0.6  # 中字体大小
FONT_THICKNESS = 2  # 字体粗细
# 新增全局字体颜色变量 (BGR格式)
COLOR_A4_INFO = (0, 0, 255)  # A4纸信息颜色(青)
COLOR_MODE_INFO = (0, 0, 255)  # 模式信息颜色(绿)
COLOR_MEASUREMENT_PROGRESS = (0, 0, 255)  # 测量进度颜色(青)
COLOR_MEASUREMENT_RESULT = (0, 0, 255)  # 测量结果颜色(绿)
COLOR_FPS = (0, 0, 255)  # FPS颜色(青)
COLOR_EDGE_LENGTH = (0, 0, 255)  # 边长度颜色(黄)
COLOR_CIRCLE_DIAMETER = (0, 0, 255)  # 圆直径颜色(绿)

# 测量参数
MEASUREMENT_DURATION = 4  # 测量持续时间(秒)
OUTLIER_REMOVE_RATIO = 0.2  # 移除首尾异常值的比例(0-0.5)

# 全局测量状态变量
is_measuring = False  # 是否正在测量
measure_start_time = 0  # 测量开始时间
measurement_data = []  # 存储尺寸测量数据
distance_measurement_data = []  # 新增：存储距离测量数据
measurement_result = None  # 尺寸测量结果
distance_measurement_result = None  # 新增：距离测量结果
current_shape = "Unknown"  # 当前检测到的形状


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)  # 高斯模糊去除高频噪声
    gray = cv2.bilateralFilter(gray, *BILATERAL_FILTER_PARAMS)  # 双边滤波保持边缘
    threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, *ADAPTIVE_THRESH_PARAMS)  # 自适应阈值化
    enhanced_image = cv2.equalizeHist(threshold_img)  # 直方图均衡化增强对比度
    edges = cv2.Canny(enhanced_image, *CANNY_THRESHOLDS)  # Canny边缘检测
    return edges, enhanced_image


def is_rectangle(corners):
    """验证四边形是否为矩形（通过对边是否近似相等判断）"""
    if len(corners) != 4:
        return False
    sides = [np.hypot(corners[(i+1)%4][0]-corners[i][0], corners[(i+1)%4][1]-corners[i][1]) for i in range(4)]
    return (abs(sides[0] - sides[2]) < RECTANGLE_ERROR_TOLERANCE * (sides[0] + sides[2])/2 and 
            abs(sides[1] - sides[3]) < RECTANGLE_ERROR_TOLERANCE * (sides[1] + sides[3])/2)


def calculate_scale(pixel_width, pixel_height):
    """计算像素比例和距离"""
    if pixel_width <= 0 or pixel_height <= 0:
        return 0.0, 0.0  # 比例和距离均返回0
    
    # 计算像素/厘米比例（用于后续尺寸测量）
    scale_x = pixel_width / A4_WIDTH  # 每厘米包含多少像素
    scale_y = pixel_height / A4_HEIGHT
    avg_scale = (scale_x + scale_y) / 2  # 平均比例
    
    # 计算距离
    distance_x = (A4_WIDTH * F) / pixel_width  # 基于宽度的距离（厘米）
    distance_y = (A4_HEIGHT * F) / pixel_height  # 基于高度的距离（厘米）
    avg_distance = (distance_x + distance_y) / 2  # 平均距离（厘米）
    
    return avg_scale, avg_distance  # 返回比例和距离（单位：厘米）


def detect_whiteA4(img, edges):
    frame = img.copy()
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    rect_image = None
    all_widths = []
    all_heights = []
    roi_position = (0, 0)
    scale = 0.0
    distance = 0.0  # 距离变量
    
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = [corner[0] for corner in approx]
            if is_rectangle(corners):
                x, y, w, h = cv2.boundingRect(approx)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                rect_image = img[y1:y2, x1:x2].copy()
                roi_position = (x1, y1)
                
                # 计算A4纸各边长度（像素）
                side_lengths = []
                for i in range(4):
                    x1_corner, y1_corner = corners[i]
                    x2_corner, y2_corner = corners[(i+1)%4]
                    length = int(math.hypot(x2_corner - x1_corner, y2_corner - y1_corner))
                    side_lengths.append(length)
                    # 绘制A4纸角点和边
                    cv2.circle(frame, (x1_corner, y1_corner), 4, (0, 0, 255), -1)
                    cv2.line(frame, (x1_corner, y1_corner), (x2_corner, y2_corner), (255, 255, 0), 2)
                
                # 计算平均宽度和高度（像素）
                height1, height2 = side_lengths[0], side_lengths[2]
                width1, width2 = side_lengths[1], side_lengths[3]
                avg_width = (width1 + width2) / 2
                avg_height = (height1 + height2) / 2
                all_widths.append(avg_width)
                all_heights.append(avg_height)
    
    # 显示A4纸信息（带标题，使用全局颜色）
    if all_widths and all_heights:
        total_avg_width = sum(all_widths) / len(all_widths)
        total_avg_height = sum(all_heights) / len(all_heights)
        cv2.putText(frame, f"Avg_width: {total_avg_width:.1f}px", (20, 30), 
                    FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)  # 平均宽度
        cv2.putText(frame, f"Avg_height: {total_avg_height:.1f}px", (20, 55), 
                    FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)  # 平均高度
        
        scale, distance = calculate_scale(total_avg_width, total_avg_height)
        cv2.putText(frame, f"Scale: {scale:.1f}px/cm", (20, 80),  # 像素比例
                    FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
        cv2.putText(frame, f"Distance: {distance:.1f}cm", (20, 105),  # 距离
                    FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
    
    if rect_image is None:
        rect_image = img.copy()
    return frame, rect_image, roi_position, scale, distance  # 新增返回distance


def process_roi(roi_edges):
    result = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_ROI_CONTOUR_AREA:
            cv2.drawContours(result, [cnt], -1, (255, 0, 0), 2)
            valid_contours.append(cnt)
    return result, valid_contours


def get_circle_diameter(contour, scale):
    """计算圆的直径（厘米）"""
    if scale <= 0:
        return 0.0
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter_pixel = radius * 2  # 直径 = 半径 * 2
    return diameter_pixel / scale  # 转换为厘米


def collect_measurement_data(contour, scale, shape, distance):
    """收集测量数据（尺寸+距离）"""
    global measurement_data, distance_measurement_data, current_shape
    current_shape = shape
    
    # 收集尺寸数据
    if shape == "Circle":
        diameter = get_circle_diameter(contour, scale)
        if diameter > 0:
            measurement_data.append(round(diameter, 2))
    else:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
        if len(approx) >= 3:
            max_length = 0
            for i in range(len(approx)):
                x1, y1 = approx[i][0]
                x2, y2 = approx[(i+1) % len(approx)][0]
                length_pixel = math.hypot(x2 - x1, y2 - y1)
                length_cm = length_pixel / scale if scale > 0 else 0
                if length_cm > max_length:
                    max_length = length_cm
            if max_length > 0:
                measurement_data.append(round(max_length, 2))
    
    # 新增：收集距离数据
    if distance > 0:
        distance_measurement_data.append(round(distance, 2))


def process_measurement_data(data):
    """处理测量数据（通用方法）"""
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    remove_count = int(n * OUTLIER_REMOVE_RATIO)
    if remove_count > 0:
        filtered_data = sorted_data[remove_count:-remove_count]
        if not filtered_data:
            filtered_data = sorted_data
    else:
        filtered_data = sorted_data
    m = len(filtered_data)
    if m % 2 == 1:
        median = filtered_data[m // 2]
    else:
        median = (filtered_data[m // 2 - 1] + filtered_data[m // 2]) / 2
    average = sum(filtered_data) / m
    return round((median + average) / 2, 2)


def start_measurement():
    """启动测量流程（清空尺寸和距离数据）"""
    global is_measuring, measure_start_time, measurement_data, measurement_result
    global distance_measurement_data, distance_measurement_result
    is_measuring = True
    measure_start_time = time.time()
    measurement_data = []  # 清空尺寸数据
    distance_measurement_data = []  # 清空距离数据
    measurement_result = None
    distance_measurement_result = None
    print("开始测量...")


def draw_contours_on_frame(frame_bgr, roi_contours, roi_pos, mode, scale, distance):
    x_offset, y_offset = roi_pos
    edge_info = []
    main_contour = None  # 主轮廓
    shape = "Unknown"  # 形状
    
    for cnt in roi_contours:
        cnt_in_frame = cnt + [x_offset, y_offset]
        cv2.drawContours(frame_bgr, [cnt_in_frame], -1, (0, 0, 255), 2)
        main_contour = cnt  # 简单模式下的主轮廓
        
        # 计算质心
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            frame_cX = cX + x_offset
            frame_cY = cY + y_offset
            cv2.circle(frame_bgr, (frame_cX, frame_cY), 5, (0, 0, 255), -1)
        
        # 形状识别
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON * perimeter, True)
        num_points = len(approx)
        
        if mode == "simple":
            # 形状判断
            if num_points == 3:
                shape = "Triangle"
            elif num_points == 4:
                shape = "Square"
            elif num_points > 4:
                area = cv2.contourArea(cnt)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > CIRCULARITY_THRESHOLD:
                    shape = "Circle"
            
            # 轮廓附近显示尺寸（仅数字和单位）
            if shape == "Circle" and scale > 0:
                diameter = get_circle_diameter(cnt, scale)
                cv2.putText(frame_bgr, f"{diameter:.2f}cm", 
                            (frame_cX + 10, frame_cY - 10), 
                            FONT, FONT_SIZE_MEDIUM, COLOR_CIRCLE_DIAMETER, FONT_THICKNESS)
            elif scale > 0 and num_points >= 3:
                # 多边形显示各边长度
                for i in range(min(num_points, DISPLAY_EDGE_LIMIT)):
                    x1, y1 = approx[i][0]
                    x2, y2 = approx[(i+1) % num_points][0]
                    frame_x1, frame_y1 = x1 + x_offset, y1 + y_offset
                    frame_x2, frame_y2 = x2 + x_offset, y2 + y_offset
                    pixel_length = math.hypot(x2 - x1, y2 - y1)
                    actual_length = pixel_length / scale
                    mid_x = int((frame_x1 + frame_x2) / 2)
                    mid_y = int((frame_y1 + frame_y2) / 2)
                    cv2.putText(frame_bgr, f"{actual_length:.2f}cm", 
                                (mid_x - 20, mid_y), 
                                FONT, FONT_SIZE_SMALL, COLOR_EDGE_LENGTH, FONT_THICKNESS)
                    edge_info.append(actual_length)
    
    # 测量逻辑处理
    global is_measuring, measure_start_time, measurement_result, distance_measurement_result
    if is_measuring and mode == "simple" and main_contour is not None:
        elapsed = time.time() - measure_start_time
        if elapsed <= MEASUREMENT_DURATION:
            # 显示测量进度
            progress = min(100, int((elapsed / MEASUREMENT_DURATION) * 100))
            cv2.putText(frame_bgr, f"Measuring: {progress}%", 
                        (20, 180), 
                        FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_PROGRESS, FONT_THICKNESS)
            # 收集尺寸和距离数据
            collect_measurement_data(main_contour, scale, shape, distance)
        else:
            # 测量结束，处理结果
            is_measuring = False
            measurement_result = process_measurement_data(measurement_data)
            distance_measurement_result = process_measurement_data(distance_measurement_data)  # 处理距离结果
            print(f"测量完成: 尺寸={measurement_result}cm, 距离={distance_measurement_result}cm")
    
    # 显示测量结果（尺寸+距离）
    result_y = 205  # 结果起始Y坐标
    if measurement_result is not None:
        unit = "φ" if current_shape == "Circle" else "L"
        cv2.putText(frame_bgr, f"{unit}: {measurement_result}cm", 
                    (20, result_y), 
                    FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
    
    if distance_measurement_result is not None:
        cv2.putText(frame_bgr, f"Dist: {distance_measurement_result}cm", 
                    (20, result_y + 25),  # 距离结果在尺寸结果下方
                    FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
    
    # 显示模式信息
    cv2.putText(frame_bgr, f"Mode: {mode}", 
                (20, 130), 
                FONT, FONT_SIZE_MEDIUM, COLOR_MODE_INFO, FONT_THICKNESS)
    
    return frame_bgr


def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    return current_time, fps


# 主程序
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (720, 1280)})
picam2.configure(config)
picam2.start()

current_mode = "simple"
prev_time = time.time()

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    edges, threshold_img = process(frame_bgr)
    # 新增接收distance返回值
    frame_bgr, roi, roi_pos, scale, distance = detect_whiteA4(frame, edges)
    roi_edges, shreshold_roi = process(roi)
    processed_roi, roi_contours = process_roi(roi_edges)
    

    if current_mode == "simple" and len(roi_contours) > 0:
        roi_contours = [sorted(roi_contours, key=cv2.contourArea, reverse=True)[0]]

    # 传入distance参数
    frame_bgr = draw_contours_on_frame(frame_bgr, roi_contours, roi_pos, current_mode, scale, distance)
    
    # 显示FPS
    prev_time, fps = calculate_fps(prev_time)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", 
                (20, 255),  # 调整FPS位置
                FONT, FONT_SIZE_MEDIUM, COLOR_FPS, FONT_THICKNESS)

    cv2.imshow("frame", frame_bgr)
    cv2.imshow("roi", processed_roi)
    
    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d') or key == ord('D'):
        current_mode = "complex" if current_mode == "simple" else "simple"
        print(f"切换到{current_mode}模式")
    elif key == ord('s') or key == ord('S'):
        if current_mode == "simple" and not is_measuring:
            start_measurement()

picam2.stop()
cv2.destroyAllWindows()