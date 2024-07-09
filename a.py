import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import time

cam = cv2.VideoCapture(0)
# key = 0
move_speed = 30
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # Trả về dạng dấu phẩy động float
num_ball = 15
ball_rad = 15
mStart = 49
mEnd = 68
laser_line_height = 15
mark = 0

# Thêm năng lượng
max_mana = 100
current_mana = max_mana
mana_cost = 10  # Tiêu tốn 10 mana mỗi lần há miệng
mana_regen_rate = 2  # Hồi phục 2 mana mỗi giây
last_regen_time = time.time()

ball_x = np.random.randint(0, frame_width, num_ball)
print(ball_x)
ball_y = np.random.randint(-1000, 0, num_ball)
print(ball_y)

detect_face = dlib.get_frontal_face_detector()
detect_mouth = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mouth_aspect_ratio(lmouth):
    A = dist.euclidean(lmouth[2], lmouth[10])  # 51, 59
    B = dist.euclidean(lmouth[4], lmouth[8])  # 53, 57
    C = dist.euclidean(lmouth[0], lmouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar, lmouth[0]
def move_add_ball(ball_x, ball_y, ball_to_remove):
    ball_x = np.delete(ball_x, ball_to_remove)
    ball_y = np.delete(ball_y, ball_to_remove)

    new_ball_x = np.random.randint(0, frame_width, len(ball_to_remove[ball_to_remove]))
    new_ball_y = np.random.randint(-1000, 0, len(ball_to_remove[ball_to_remove]))

    ball_x = np.concatenate((ball_x, new_ball_x))
    ball_y = np.concatenate((ball_y, new_ball_y))
    return ball_x, ball_y

while True:

    ret, frame = cam.read()
    if ret:
        move_y = np.random.randint(0, move_speed, num_ball)
        ball_y = ball_y + move_y

        ball_to_remove = ball_y > frame.shape[0]
        ball_x, ball_y = move_add_ball(ball_x, ball_y, ball_to_remove)

        for i in range(0, num_ball):
            cv2.circle(frame, (ball_x[i], ball_y[i]), radius=ball_rad, color=(0, 255, 0), thickness=-1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detect_face(gray, 0)

        for rect in rects:
            shape = detect_mouth(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]  # Facial landmark

            mouth_ratio, mouth_position = mouth_aspect_ratio(mouth)  # mouth: 1 mảng các tọa độ của miệng

            if (mouth_ratio > 0.7) & (current_mana >= mana_cost):
                current_mana -= mana_cost
                laser_line_y = mouth_position[1] # 0 la x, 1 la y. Nếu dùng y thì vẽ đường ngang, dùng x thì vẽ đường dọc


                cv2.rectangle(frame, (0, laser_line_y),
                              (frame_width, laser_line_y + laser_line_height), (0, 0, 255), thickness=-1)

                ball_to_remove = (ball_y >= laser_line_y) & (ball_y <= laser_line_y + laser_line_height)
                ball_x, ball_y = move_add_ball(ball_x, ball_y, ball_to_remove)

                # Hiển thị điểm lên màn hình
                mark = mark + len(ball_to_remove[ball_to_remove])

        # Hồi phục mana theo time
        current_time = time.time()
        if current_time - last_regen_time >= 1:  # Mỗi giây hồi phục mana
            current_mana = min(current_mana + mana_regen_rate, max_mana)
            last_regen_time = current_time

        # Hiển thị điểm và mana lên màn hình
        cv2.putText(frame, "Points: " + str(mark), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Mana: " + str(current_mana), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Game", frame)


    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()