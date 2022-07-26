import os
import random
import cv2

from right import RightPost
import math
from ball_detect import get_circle, get_ball_box

def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1 + 1e-8)


def calculate_vertical_line(line):
    k1 = calculate_slope(line)
    k2 = -1 / k1
    return k2


def calculate_point_to_line(k, b, point):
    x0, y0 = point
    return abs(k*x0-y0+b) / math.sqrt(k**2+1)


def judge(ball_center, ball_radium, line, box, right=True):
    if not ball_center:
        return False
    k1 = calculate_slope(line)
    k2 = calculate_vertical_line(line)
    b1 = line[0][1] - k1*line[0][0]
    real_ball_x = ball_center[0] + box[0]
    real_ball_y = ball_center[1] + box[1]
    b2 = real_ball_y - k2 * real_ball_x
    cross_point_x = (b2-b1) / (k1-k2)
    distance = calculate_point_to_line(k1, b1, [real_ball_x, real_ball_y])
    if distance > ball_radium and (cross_point_x > real_ball_x if right else cross_point_x < ball_center[0]):
        return True
    else:
        return False


def goal_change(last, this):
    if last is False and this is True:
        return True
    else:
        return False


def record10s(img_array, not_goal_list, goal_list, is_GOAL, goal_recorded):
    if (is_GOAL is False) and (goal_recorded is False):
        not_goal_list.append(img_array)
        if len(not_goal_list) > 75:
            not_goal_list.pop(0)
            return False
    else:
        goal_list.append(img_array)
        if len(goal_list) >= 75:
            record = not_goal_list.extend(goal_list)
            make_video(record)
            goal_list.clear()
            return False
        return True


def make_video(record):
    fps = 15 
    video = cv2.VideoWriter('goal' + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                            (320, 240))
    for image in record:
            video.write(image)
    video.release()
    
    
if __name__ == '__main__':

    cap = cv2.VideoCapture('videos/right.avi')
    not_goal_list = []
    goal_list = []
    this_goal = False
    goal_recorded = False
    rightpost = RightPost()
    while True:
        last_goal = this_goal
        ret, img = cap.read()
        if not ret:
            break
        img_copy = img.copy()
        rightpost.get_lines(img)
    
        x1, y1, x2, y2 = get_ball_box(img_copy)
        if x1 or y1 or x2 or y2:
            print(x1,y1,x2,y2)
            ball = img_copy[y1:y2,x1:x2]
            print(ball.shape)
            circle_center, radium = get_circle(rightpost, ball)
            if radium:
                cv2.circle(rightpost.color_img, (x1+circle_center[0],y1+circle_center[1]), radium, (0,0,255), 2)
        else:
            circle_center, radium = None, None

        goal = judge(circle_center,radium,rightpost.inner,[x1,y1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        if goal:
         
            cv2.putText(rightpost.color_img, 'GOAL', (0,220), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(rightpost.color_img, 'NO GOAL', (0,220), font, 1, (255, 0, 0), 2)

        rightpost.show_img(rightpost.color_img,1)
        
        this_goal = goal
        is_GOAL = goal_change(last_goal,this_goal)
        if (is_GOAL is False) and (goal_recorded is False):
            not_goal_list.append(img)
            if len(not_goal_list) > 15:
                not_goal_list.pop(0)
        else:
            goal_recorded = True
            goal_list.append(img)
            if len(goal_list) >= 45:
                record = not_goal_list
                record.extend(goal_list)
                print(record)
                make_video(record)
                goal_list.clear()
                goal_recorded = False
            
         


