
import cv2
import numpy as np
import time
import math
import libcamera
from picamera2 import Picamera2
import serial
from threading import Thread, Lock

# 确保中文显示正常（OpenCV不直接支持中文，此处用默认字体）
def nothing(x):
    pass

# 创建窗口和轨迹栏
cv2.namedWindow("Color Mast", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("l1", "Color Mast", 1906, 2000, nothing)


# 物理尺寸参数
A4_WIDTH = 17.3  # A4纸实际宽度(单位:厘米)
A4_HEIGHT = 25.8  # A4纸实际高度(单位:厘米)
# 图像处理参数
GAUSSIAN_KERNEL = (5, 5)  # 高斯模糊核大小
BILATERAL_FILTER_PARAMS = (9, 75, 75)  # 双边滤波参数
ADAPTIVE_THRESH_PARAMS = (11, 2)  # 自适应阈值参数(块大小, 常数)
CANNY_THRESHOLDS = (50, 150)  # Canny边缘检测阈值
# 轮廓检测参数
MIN_CONTOUR_AREA = 20000  # 最小轮廓面积(过滤噪声)
MIN_ROI_CONTOUR_AREA = 1500  # ROI区域最小轮廓面积
APPROX_POLY_EPSILON = 0.03  # 多边形逼近系数
RECTANGLE_ERROR_TOLERANCE = 0.03  # 矩形判断容差
CIRCULARITY_THRESHOLD = 0.7  # 圆形判断阈值
SQUARE_ASPECT_RATIO_TOLERANCE = 0.15  # 正方形宽高比容差
# 直角与等长检测参数
RIGHT_ANGLE_TOLERANCE = 30  # 直角容忍度(度)
EQUAL_LENGTH_TOLERANCE = 8.0  # 边长相等容忍度(像素)
# 显示参数
DISPLAY_EDGE_LIMIT = 5  # 最大显示边数量
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
FONT_SIZE_SMALL = 0.5  # 小字体大小
FONT_SIZE_MEDIUM = 0.6  # 中字体大小
FONT_THICKNESS = 2  # 字体粗细
# 全局字体颜色变量 (BGR格式)
COLOR_A4_INFO = (255, 0, 0)  # A4纸信息颜色(蓝)
COLOR_MODE_INFO = (0, 255, 0)  # 模式信息颜色(绿)
COLOR_MEASUREMENT_PROGRESS = (255, 255, 0)  # 测量进度颜色(青)
COLOR_MEASUREMENT_RESULT = (0, 255, 0)  # 测量结果颜色(绿)
COLOR_FPS = (255, 255, 0)  # FPS颜色(青)
COLOR_EDGE_LENGTH = (0, 255, 255)  # 边长度颜色(黄)
COLOR_CIRCLE_DIAMETER = (0, 255, 0)  # 圆直径颜色(绿)
COLOR_CORNER_POINT = (0, 0, 255)  # 角点颜色(红)
COLOR_ERROR = (0, 0, 255)  # 错误提示颜色(红)
COLOR_SELECTED_SQUARE = (0, 165, 255)  # 选中的正方形颜色(橙)
# 测量参数
MEASUREMENT_DURATION = 2.5  # 测量持续时间(秒)
OUTLIER_REMOVE_RATIO = 0.2  # 移除首尾异常值的比例(0-0.5)

# 测量状态变量（全局变量）
is_measuring = False  # 是否正在测量
measure_start_time = 0  # 测量开始时间
square_measurements = []  # 存储所有正方形的边长数据
all_square_sides = []  # 存储所有检测到的正方形边长（保留历史）
measurement_result = None  # 最终测量结果（最小值）
result_sent = False  # 结果是否已发送
current_measured_shape = "Unknown"  # 当前测量的形状
selected_square_index = -1  # 选中要测量的正方形索引（1-5）

D = None  # 实际距离(厘米)
L = None  # 最小正方形边长(厘米)
current_mode = "simple"  # 全局模式变量：simple, complex
mode_lock = Lock()  # 模式切换锁，确保线程安全


def process(img):
    """图像预处理：灰度化、滤波、阈值化和边缘检测"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    gray = cv2.bilateralFilter(gray, *BILATERAL_FILTER_PARAMS)
    threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, *ADAPTIVE_THRESH_PARAMS)
    edges = cv2.Canny(threshold_img, *CANNY_THRESHOLDS)
    return edges, threshold_img


def is_rectangle(corners):
    """验证四边形是否为矩形（通过对边是否近似相等判断）"""
    if len(corners) != 4:
        return False
    sides = [np.hypot(corners[(i+1)%4][0]-corners[i][0], corners[(i+1)%4][1]-corners[i][1]) 
            for i in range(4)]
    return (abs(sides[0] - sides[2]) < RECTANGLE_ERROR_TOLERANCE * (sides[0] + sides[2])/2 and 
            abs(sides[1] - sides[3]) < RECTANGLE_ERROR_TOLERANCE * (sides[1] + sides[3])/2)


def is_square(contour):
    """判断轮廓是否为正方形"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
    
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False, None
    
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    
    if 1 - SQUARE_ASPECT_RATIO_TOLERANCE <= aspect_ratio <= 1 + SQUARE_ASPECT_RATIO_TOLERANCE:
        return True, approx
    
    return False, None


def get_contour_centroid(contour):
    """计算轮廓的质心坐标"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return (0, 0)


def sort_squares_by_centroid(squares):
    """按质心从上到下（y坐标从小到大）排序正方形"""
    # 为每个正方形计算质心
    squares_with_centroids = []
    for cnt in squares:
        centroid = get_contour_centroid(cnt)
        squares_with_centroids.append((cnt, centroid))
    
    # 按y坐标排序（y值越小越靠上）
    squares_with_centroids.sort(key=lambda x: x[1][1])
    
    # 返回排序后的轮廓列表
    return [item[0] for item in squares_with_centroids]


def get_circle_diameter(contour, scale):
    """计算圆的直径（厘米）"""
    if scale <= 0:
        return 0.0
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return (radius * 2) / scale  # 直径 = 半径 * 2


def get_square_side_length(contour, scale):
    """计算正方形边长（厘米）"""
    if scale <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
    if len(approx) != 4:
        return 0.0
    
    sides = []
    for i in range(4):
        x1, y1 = approx[i][0]
        x2, y2 = approx[(i+1)%4][0]
        sides.append(math.hypot(x2-x1, y2-y1))
    return sum(sides)/4 / scale


def get_square_distance_from_a4(square_contour, a4_y, scale):
    """计算正方形到A4纸顶部的距离（厘米）"""
    if scale <= 0:
        return 0.0
    
    # 获取正方形的顶部y坐标
    x, y, w, h = cv2.boundingRect(square_contour)
    square_top_y = y
    
    # 计算像素距离
    pixel_distance = square_top_y - a4_y
    
    # 转换为实际距离（厘米）
    return abs(pixel_distance) / scale


def get_all_squares_and_measure(squares, scale, a4_y=0):
    """获取所有正方形并返回其边长和距离A4纸顶部的距离"""
    square_data = []
    for cnt in squares:
        side_length = get_square_side_length(cnt, scale)
        distance = get_square_distance_from_a4(cnt, a4_y, scale)
        if side_length > 0:
            square_data.append((cnt, side_length, distance))
    return square_data


def find_smallest_square(contours):
    """从轮廓列表中找到最小的正方形"""
    smallest_square = None
    smallest_area = float('inf')
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        is_sq, approx = is_square(cnt)
        
        if is_sq and area < smallest_area:
            smallest_area = area
            smallest_square = cnt
            
    return smallest_square


def get_polygon_corners(contour):
    """获取多边形的角点"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.008 * perimeter, True)
    corners = [corner[0] for corner in approx]
    return corners, approx


def distance_between_points(p1, p2):
    """计算两点间距离"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_angle(a, b, c):
    """计算三点形成的角度（角ABC）"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    len_ba = math.hypot(ba[0], ba[1])
    len_bc = math.hypot(bc[0], bc[1])
    
    if len_ba < 1e-6 or len_bc < 1e-6:
        return 0.0
    
    cos_angle = dot_product / (len_ba * len_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


def calculate_square_point(a, b, c):
    """计算正方形的第四个点D"""
    ba = (a[0] - b[0], a[1] - b[1])
    return (c[0] + ba[0], c[1] + ba[1])


def calculate_scale(pixel_width, pixel_height):
    """计算像素/厘米比例和拍摄距离"""
    if pixel_width <= 0 or pixel_height <= 0:
        return 0.0, 0.0
    
    scale_x = pixel_width / A4_WIDTH
    scale_y = pixel_height / A4_HEIGHT
    avg_scale = (scale_x + scale_y) / 2
    
    F = cv2.getTrackbarPos("l1", "Color Mast")
    distance_x = (A4_WIDTH * F) / pixel_width if pixel_width != 0 else 0
    distance_y = (A4_HEIGHT * F) / pixel_height if pixel_height != 0 else 0
    avg_distance = (distance_x + distance_y) / 2 if (distance_x + distance_y) != 0 else 0
    
    return avg_scale, avg_distance


def detect_whiteA4(img, edges):
    """检测A4纸并计算比例和距离"""
    frame = img.copy()
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    rect_image = None
    all_widths = []
    all_heights = []
    roi_position = (0, 0)
    scale = 0.0
    distance = 0.0
    a4_detected = False
    a4_top_y = 0  # A4纸顶部y坐标
    
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 500:
            continue
            
        approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = [corner[0] for corner in approx]
            if is_rectangle(corners):
                x, y, w, h = cv2.boundingRect(approx)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                
                # 记录A4纸顶部y坐标
                a4_top_y = y1
                
                rect_image = img[y1+2:y2-2, x1+2:x2-2].copy()
                roi_position = (x1+2, y1+2)
                
                side_lengths = []
                for i in range(4):
                    x1_corner, y1_corner = corners[i]
                    x2_corner, y2_corner = corners[(i+1)%4]
                    length = int(math.hypot(x2_corner - x1_corner, y2_corner - y1_corner))
                    side_lengths.append(length)
                    cv2.circle(frame, (x1_corner, y1_corner), 4, (0, 0, 255), -1)
                    cv2.line(frame, (x1_corner, y1_corner), (x2_corner, y2_corner), (255, 255, 0), 2)
                
                height1, height2 = side_lengths[0], side_lengths[2]
                width1, width2 = side_lengths[1], side_lengths[3]
                avg_width = (width1 + width2) / 2
                avg_height = (height1 + height2) / 2
                
                all_widths.append(avg_width)
                all_heights.append(avg_height)
                a4_detected = True

    # 处理A4纸检测结果
    if a4_detected and all_widths and all_heights:
        total_avg_width = sum(all_widths) / len(all_widths)
        total_avg_height = sum(all_heights) / len(all_heights)
        
        cv2.putText(frame, f"Avg Width: {total_avg_width:.1f}px", (20, 30), 
                   FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
        cv2.putText(frame, f"Avg Height: {total_avg_height:.1f}px", (20, 55), 
                   FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
        
        scale, distance = calculate_scale(total_avg_width, total_avg_height)
        cv2.putText(frame, f"Scale: {scale:.1f}px/cm", (20, 80), 
                   FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
        cv2.putText(frame, f"Distance: {distance:.1f}cm", (20, 105), 
                   FONT, FONT_SIZE_MEDIUM, COLOR_A4_INFO, FONT_THICKNESS)
        
        global D
        D = round(distance, 2)
    else:
        cv2.putText(frame, "No A4", (20, 130), 
                   FONT, FONT_SIZE_MEDIUM, COLOR_ERROR, FONT_THICKNESS)

    if rect_image is None:
        rect_image = img.copy()
        
    return frame, rect_image, roi_position, scale, distance, a4_top_y


def process_roi(roi_edges):
    """处理ROI区域，提取物体轮廓"""
    result = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_ROI_CONTOUR_AREA:
            cv2.drawContours(result, [cnt], -1, (255, 0, 0), 2)
            valid_contours.append(cnt)
            
    return result, valid_contours


def process_measurement_data(data):
    """过滤异常值并计算最终结果 - 修复了元组比较问题"""
    if not data:
        return None
    
    # 按正方形序号分组
    grouped_data = {}
    for item in data:
        idx, side, dist = item
        if idx not in grouped_data:
            grouped_data[idx] = []
        grouped_data[idx].append((side, dist))
    
    # 对每组数据分别排序和过滤
    filtered_data = []
    for idx, measurements in grouped_data.items():
        # 按边长排序
        sorted_measurements = sorted(measurements, key=lambda x: x[0])
        n = len(sorted_measurements)
        remove_count = int(n * OUTLIER_REMOVE_RATIO)
        
        if remove_count > 0:
            group_filtered = sorted_measurements[remove_count:-remove_count]
            if not group_filtered:
                group_filtered = sorted_measurements
        else:
            group_filtered = sorted_measurements
            
        # 将过滤后的数据添加回结果列表，包含序号
        for side, dist in group_filtered:
            filtered_data.append((idx, side, dist))
            
    return filtered_data


def draw_contours_on_frame(frame_bgr, roi_contours, roi_pos, mode, scale, distance, a4_top_y):
    """在帧上绘制轮廓和测量结果"""
    global is_measuring, measure_start_time, measurement_result, square_measurements
    global all_square_sides, current_measured_shape, L, result_sent, selected_square_index
    
    x_offset, y_offset = roi_pos
    main_contour = None
    shape = "Unknown"
    squares = []
    other_contours = []
    corner_coordinates = []
    sorted_squares = []  # 存储排序后的正方形
    
    # 获取主轮廓
    if roi_contours:
        if mode == "simple":
            main_contour = max(roi_contours, key=lambda c: cv2.contourArea(c))
        else:  # complex mode
            smallest_square = find_smallest_square(roi_contours)
            main_contour = smallest_square if smallest_square is not None else \
                          max(roi_contours, key=lambda c: cv2.contourArea(c))

    # 简单模式处理
    if mode == "simple":
        for cnt in roi_contours:
            cnt_in_frame = cnt + [x_offset, y_offset]
            cv2.drawContours(frame_bgr, [cnt_in_frame], -1, (0, 0, 255), 2)
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                frame_cX = cX + x_offset
                frame_cY = cY + y_offset
                cv2.circle(frame_bgr, (frame_cX, frame_cY), 5, (0, 0, 255), -1)
                
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON * perimeter, True)
            num_points = len(approx)
            
            if num_points == 3:
                shape = "Triangle"
            elif num_points == 4:
                shape = "Square"
            elif num_points > 4:
                area = cv2.contourArea(cnt)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity > CIRCULARITY_THRESHOLD:
                    shape = "Circle"
            
            # 显示尺寸
            if shape == "Circle" and scale > 0:
                diameter = get_circle_diameter(cnt, scale)
                cv2.putText(frame_bgr, f"{diameter:.2f}cm", 
                           (frame_cX + 10, frame_cY - 10), 
                           FONT, FONT_SIZE_MEDIUM, COLOR_CIRCLE_DIAMETER, FONT_THICKNESS)
                if is_measuring:
                    square_measurements.append((-1, diameter, 0))  # 使用-1表示圆形直径
            elif scale > 0 and num_points >= 3:
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
                    if is_measuring:
                        square_measurements.append((-2, actual_length, 0))  # 使用-2表示多边形边长

        # 复杂模式处理
    elif mode == "complex":
        # 区分正方形和其他轮廓
        for cnt in roi_contours:
            is_sq, _ = is_square(cnt)
            if is_sq:
                squares.append(cnt)
            else:
                other_contours.append(cnt)

        # 处理检测到的正方形
        if len(squares) > 0:
            # 按质心从上到下排序正方形
            sorted_squares = sort_squares_by_centroid(squares)
            
            if len(sorted_squares) == 1:
                shape = "Square"
                main_contour = sorted_squares[0]
                cnt_in_frame = main_contour + [x_offset, y_offset]
                cv2.drawContours(frame_bgr, [cnt_in_frame], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt_in_frame)
                cv2.putText(frame_bgr, "Square 1", (x + 10, y - 10),
                           FONT, FONT_SIZE_MEDIUM, (0, 255, 0), FONT_THICKNESS)
                
                # 计算并存储边长
                side_length = get_square_side_length(main_contour, scale)
                distance_from_a4 = get_square_distance_from_a4(main_contour, a4_top_y, scale)
                if is_measuring and side_length > 0:
                    square_measurements.append((1, side_length, distance_from_a4))
            else:
                shape = "Multiple Squares"
                # 获取所有正方形及其边长和距离
                square_data = get_all_squares_and_measure(sorted_squares, scale, a4_top_y)
                
                # 绘制所有正方形并标记序号
                for i, (cnt, side_length, distance_from_a4) in enumerate(square_data):
                    cnt_in_frame = cnt + [x_offset, y_offset]
                    # 如果是当前选中要测量的正方形，用不同颜色标记
                    if is_measuring and selected_square_index == i + 1:
                        cv2.drawContours(frame_bgr, [cnt_in_frame], -1, COLOR_SELECTED_SQUARE, 3)
                    else:
                        cv2.drawContours(frame_bgr, [cnt_in_frame], -1, (0, 255, 0), 2)
                        
                    x, y, w, h = cv2.boundingRect(cnt_in_frame)
                    cv2.putText(frame_bgr, f"Square {i+1}: {side_length:.2f}cm, Dist: {distance_from_a4:.2f}cm", 
                               (x + 10, y - 10), FONT, FONT_SIZE_SMALL, (0, 255, 0), 1)
                    
                    # 存储所有正方形数据，特别是选中的正方形
                    if is_measuring:
                        if selected_square_index == -1 or selected_square_index == i + 1:
                            square_measurements.append((i + 1, side_length, distance_from_a4))
                
                # 标记最小正方形
                main_contour = find_smallest_square(sorted_squares)
                if main_contour is not None:
                    cnt_in_frame = main_contour + [x_offset, y_offset]
                    cv2.drawContours(frame_bgr, [cnt_in_frame], -1, (0, 0, 255), 3)  # 红色标记最小正方形
                
        # 处理其他轮廓（多边形等）
        elif len(other_contours) > 0:
            main_contour = max(other_contours, key=lambda c: cv2.contourArea(c))
            corners, approx = get_polygon_corners(main_contour)
            num_corners = len(corners)
            
            # 判断形状类型
            if num_corners == 3:
                shape = "Triangle"
            elif num_corners == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 1 - SQUARE_ASPECT_RATIO_TOLERANCE <= aspect_ratio <= 1 + SQUARE_ASPECT_RATIO_TOLERANCE:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif num_corners > 6:
                shape = f"Polygon ({num_corners} sides)"
                
            # 标记角点
            corner_coordinates = [(x + x_offset, y + y_offset) for x, y in corners]
            for idx, (x, y) in enumerate(corner_coordinates):
                cv2.circle(frame_bgr, (x, y), 5, COLOR_CORNER_POINT, -1)
                cv2.putText(frame_bgr, f"{idx}", (x + 5, y - 5),
                           FONT, FONT_SIZE_SMALL, COLOR_CORNER_POINT, 1)
            
            if corner_coordinates:
                cv2.putText(frame_bgr, f"Corners: {len(corner_coordinates)}", 
                           (20, 255), FONT, FONT_SIZE_MEDIUM, COLOR_CORNER_POINT, FONT_THICKNESS)

    # 测量逻辑
    if is_measuring and main_contour is not None and scale > 0:
        elapsed = time.time() - measure_start_time
        if elapsed <= MEASUREMENT_DURATION:
            progress = min(100, int((elapsed / MEASUREMENT_DURATION) * 100))
            if selected_square_index == -1:
                cv2.putText(frame_bgr, f"Measuring... {progress}%", 
                           (20, 450), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_PROGRESS, FONT_THICKNESS)
            else:
                cv2.putText(frame_bgr, f"Measuring Square {selected_square_index}... {progress}%", 
                           (20, 450), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_PROGRESS, FONT_THICKNESS)
            
            # 针对多边形和矩形计算可能的正方形边长
            if shape.startswith("Polygon") or shape == "Rectangle":
                current_measured_shape = "ABCD Square"
                corners, _ = get_polygon_corners(main_contour)
                num_corners = len(corners)
                
                if num_corners >= 3:
                    for i in range(num_corners):
                        # 选取角点B，及其相邻角点A和C
                        b_idx = i
                        b = (corners[b_idx][0] + x_offset, corners[b_idx][1] + y_offset)
                        
                        a_idx = (i - 1) % num_corners
                        c_idx = (i + 1) % num_corners
                        a = (corners[a_idx][0] + x_offset, corners[a_idx][1] + y_offset)
                        c = (corners[c_idx][0] + x_offset, corners[c_idx][1] + y_offset)
                        
                        # 计算BA和BC的长度
                        ba_len = distance_between_points(b, a)
                        bc_len = distance_between_points(b, c)
                        
                        # 检查两边是否近似相等
                        if abs(ba_len - bc_len) > EQUAL_LENGTH_TOLERANCE:
                            continue
                        
                        # 检查角度是否近似直角
                        angle = calculate_angle(a, b, c)
                        if abs(angle - 90) < RIGHT_ANGLE_TOLERANCE:
                            # 计算正方形第四个点D
                            d = calculate_square_point(a, b, c)
                            
                            # 绘制正方形
                            cv2.line(frame_bgr, a, b, (0, 255, 255), 3)
                            cv2.line(frame_bgr, b, c, (0, 255, 255), 3)
                            cv2.line(frame_bgr, c, d, (0, 255, 255), 3)
                            cv2.line(frame_bgr, d, a, (0, 255, 255), 3)
                            
                            # 标记直角点B
                            cv2.circle(frame_bgr, b, 8, (0, 255, 0), -1)
                            cv2.putText(frame_bgr, "B", (b[0]+10, b[1]-10),
                                       FONT, FONT_SIZE_MEDIUM, (0, 255, 0), FONT_THICKNESS)
                            
                            # 计算并存储正方形边长
                            side_length = ba_len / scale if scale > 0 else 0
                            if side_length > 0:
                                cv2.putText(frame_bgr, f"Side: {side_length:.2f}cm", 
                                           (int(d[0])+10, int(d[1])+10),
                                           FONT, FONT_SIZE_MEDIUM, (255, 255, 255), FONT_THICKNESS)
                                square_measurements.append((-3, side_length, 0))  # -3表示计算出的正方形
        else:
            # 测量结束，计算结果
            is_measuring = False
            # 处理测量数据（过滤异常值）
            filtered_data = process_measurement_data(square_measurements)
            
            if filtered_data:
                # 按正方形序号分组并计算平均值
                square_results = {}
                for item in filtered_data:
                    idx, side, dist = item
                    if idx not in square_results:
                        square_results[idx] = []
                    square_results[idx].append((side, dist))
                
                # 计算每个正方形的平均边长和距离
                final_results = {}
                for idx, measurements in square_results.items():
                    sides = [m[0] for m in measurements]
                    dists = [m[1] for m in measurements]
                    avg_side = sum(sides) / len(sides)
                    avg_dist = sum(dists) / len(dists) if len(dists) > 0 else 0
                    final_results[idx] = (avg_side, avg_dist)
                
                measurement_result = final_results
                all_square_sides.append(final_results)
            else:
                measurement_result = None
                all_square_sides = []  # 清空无效数据
            
            result_sent = False  # 重置发送状态
            
            # 打印测量信息
            if measurement_result is not None and D is not None:
                print("测量完成:")
                for idx, (side, dist) in measurement_result.items():
                    if idx == -1:
                        print(f"  圆形直径: {side:.2f}cm")
                    elif idx == -2:
                        print(f"  多边形边长: {side:.2f}cm")
                    elif idx == -3:
                        print(f"  计算出的正方形: 边长 = {side:.2f}cm")
                    else:
                        print(f"  正方形 {idx}: 边长 = {side:.2f}cm, 距离A4纸 = {dist:.2f}cm")
            else:
                print("测量失败: 未检测到有效的正方形或距离数据")

    # 显示测量结果
    if measurement_result is not None:
        result_y = 500
        # 显示每个正方形的结果
        for idx, (side, dist) in measurement_result.items():
            if idx == -1:
                cv2.putText(frame_bgr, f"Circle Diameter: {side:.2f}cm", 
                           (20, result_y), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
            elif idx == -2:
                cv2.putText(frame_bgr, f"Polygon Side: {side:.2f}cm", 
                           (20, result_y), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
            elif idx == -3:
                cv2.putText(frame_bgr, f"Calculated Square: {side:.2f}cm", 
                           (20, result_y), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
            else:
                cv2.putText(frame_bgr, f"Square {idx}: L={side:.2f}cm, D={dist:.2f}cm", 
                           (20, result_y), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
            result_y += 25
        
        # 显示A4纸距离
        cv2.putText(frame_bgr, f"A4 Distance: {D if D is not None else 'N/A'}cm", 
                   (20, result_y), FONT, FONT_SIZE_MEDIUM, COLOR_MEASUREMENT_RESULT, 2)
    
    # 显示模式信息
    cv2.putText(frame_bgr, f"Mode: {mode}", (20, 130),
               FONT, FONT_SIZE_MEDIUM, COLOR_MODE_INFO, FONT_THICKNESS)
    return frame_bgr   


def start_measurement(index=-1):
    """启动测量流程，index为要测量的正方形序号，-1表示测量所有"""
    global is_measuring, measure_start_time, square_measurements, measurement_result
    global all_square_sides, result_sent, selected_square_index
    is_measuring = True
    measure_start_time = time.time()
    selected_square_index = index  # 设置要测量的正方形序号
    square_measurements = []  # 清空当前测量周期的数据
    measurement_result = None
    result_sent = False  # 重置发送状态
    if index == -1:
        print("开始测量所有正方形...")
    else:
        print(f"开始测量第 {index} 个正方形...")


# 串口通信相关函数
def init_serial(port='/dev/ttyAMA0', baudrate=9600):
    """初始化串口"""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
        if ser.isOpen():
            print(f"串口 {port} 已打开，波特率: {baudrate}")
            return ser
        else:
            print("串口打开失败")
            return None
    except Exception as e:
        print(f"串口初始化错误: {e}")
        return None


def send_data(ser, data):
    """发送数据到串口"""
    if ser and ser.isOpen():
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            ser.write(data)
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False
    return False


def receive_data(ser, max_bytes=1024):
    """从串口接收数据"""
    if ser and ser.isOpen():
        try:
            bytes_available = ser.inWaiting()
            if bytes_available > 0:
                data = ser.read(bytes_available)
                try:
                    str_data = data.decode('utf-8').strip()  # 去除空格和换行
                    print(f"接收: {str_data}")
                    return str_data
                except UnicodeDecodeError:
                    print(f"接收原始字节: {data}")
                    return None
        except Exception as e:
            print(f"接收失败: {e}")
    return None


def serial_communication_loop(ser):
    """串口通信线程循环"""
    global measurement_result, D, current_mode, is_measuring, result_sent, all_square_sides
    while True:
        # 检查是否有新的测量结果且尚未发送
        if measurement_result is not None and D is not None and not result_sent and not is_measuring:
            # 构建发送字符串，统一使用L：边长，D：距离格式
            send_str = ""
            for idx, (side, dist) in measurement_result.items():
                if idx == -1:  # 圆形（直径作为特殊的"边长"）
                    send_str += f"Cicrle:L={side:.2f}cm"
                elif idx == -2:  # 多边形边长
                    send_str += f"Muiltple:L={side:.2f}cm"
                elif idx == -3:  # 计算出的正方形
                    send_str += f"Square:L={side:.2f}cm"
                else:  # 检测到的正方形
                    send_str += f"Square{idx}:L={side:.2f}cm"
            
            # 添加A4纸距离
            send_str += f",D={D}cm\n"
            
            # 发送数据并标记为已发送
            if send_data(ser, send_str):
                result_sent = True
                print(f"测量结果已发送: {send_str.strip()}")
        
        # 接收数据并处理模式切换和测量指令
        received = receive_data(ser)
        if received:
            with mode_lock:  # 确保模式切换线程安全
                if received == "1":
                    current_mode = "simple"
                    send_data(ser, "Simple measurement mode\n")
                    start_measurement()
                elif received == "2":
                    current_mode = "complex"
                    send_data(ser, "Complex measurement mode\n")
                    start_measurement()
                elif received in ["5", "6", "7", "8", "9"]:
                    # 5-9对应测量第1-5个正方形
                    square_index = int(received) - 4
                    current_mode = "complex"
                    send_data(ser, f"Measuring square {square_index}\n")
                    start_measurement(square_index)
                elif received == "q":
                    send_data(ser, "Exiting program\n")
                    break
            
        time.sleep(0.01)


def main():
    """主函数"""
    # 初始化摄像头
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (720, 1280)})
    picam2.configure(config)
    picam2.start()
    # 初始化串口
    ser = init_serial(port='/dev/ttyAMA0', baudrate=115200)
    # 启动串口通信线程
    if ser:
        serial_thread = Thread(target=serial_communication_loop, args=(ser,), daemon=True)
        serial_thread.start()
    
    global current_mode
    prev_time = time.time()
    cv2.imshow("Color Mast", np.zeros((1, 300, 3), dtype=np.uint8))
    
    try:
        while True:
            # 采集图像
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 图像处理流程
            edges, threshold_img = process(frame_bgr)
            frame_bgr, roi, roi_pos, scale, distance, a4_top_y = detect_whiteA4(frame_bgr, edges)
            roi_edges, roi_threshold = process(roi)
            processed_roi, roi_contours = process_roi(roi_edges)
            
            # 绘制轮廓和测量结果（使用锁确保模式读取安全）
            with mode_lock:
                current_mode_copy = current_mode
            frame_bgr = draw_contours_on_frame(frame_bgr, roi_contours, roi_pos, current_mode_copy, scale, distance, a4_top_y)

            # 显示窗口
            cv2.imshow("frame", frame_bgr)
            cv2.imshow("roi", processed_roi)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # 支持键盘模式切换（调试用）
            elif key == ord('1'):
                with mode_lock:
                    current_mode = "simple"
                print("切换到简单模式")
            elif key == ord('2'):
                with mode_lock:
                    current_mode = "complex"
                print("切换到复杂模式")
            elif key in [ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
                square_index = int(chr(key)) - 4
                with mode_lock:
                    current_mode = "complex"
                print(f"测量第 {square_index} 个正方形")
                start_measurement(square_index)
            elif key == ord('s'):
                if not is_measuring:
                    start_measurement()
                    
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if ser and ser.isOpen():
            ser.close()
            print("串口已关闭")


if __name__ == "__main__":
    main()
