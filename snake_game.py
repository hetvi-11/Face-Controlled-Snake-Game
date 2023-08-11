import cv2
import dlib
import pygame
import sys
import numpy as np

# Pygame initialization
pygame.init()
win = pygame.display.set_mode((650, 650))
pygame.display.set_caption("Face Controlled Snake")

# Initialize dlib's face detector (HOG-based) and create a facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('c:\\Users\\HP\\Downloads\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

snake_pos = [[100, 50], [90, 50], [80, 50]]
snake_speed = 1
direction = 'RIGHT'
food_pos = [np.random.randint(1,50)*10, np.random.randint(1,50)*10]
food_spawn = True
movement_threshold = 10

def game_over():
    pygame.quit()
    sys.exit()

def calculate_face_center(shape):
    x = (shape.parts()[27].x + shape.parts()[28].x) // 2
    y = (shape.parts()[27].y + shape.parts()[28].y) // 2
    return (x, y)

def detect_face_movement(neutral_center, p2):
    # Calculate the current face center
    p2_center = calculate_face_center(p2)

    x_diff = p2_center[0] - neutral_center[0]
    y_diff = p2_center[1] - neutral_center[1]
    
    if abs(x_diff) > movement_threshold:
        if x_diff > 0:
            return 'RIGHT'
        else:
            return 'LEFT'
    elif abs(y_diff) > movement_threshold:
        if y_diff > 0:
            return 'DOWN'
        else:
            return 'UP'
    else:
        return None

def draw_snake(snake_pos):
    for pos in snake_pos:
        pygame.draw.rect(win, (0, 255, 0), pygame.Rect(pos[0], pos[1], 10, 10))

def update_snake_pos(snake_pos, direction):
    if direction == 'RIGHT':
        snake_pos.insert(0, list(np.add(snake_pos[0], [10, 0])))
    elif direction == 'LEFT':
        snake_pos.insert(0, list(np.add(snake_pos[0], [-10, 0])))
    elif direction == 'UP':
        snake_pos.insert(0, list(np.add(snake_pos[0], [0, -10])))
    elif direction == 'DOWN':
        snake_pos.insert(0, list(np.add(snake_pos[0], [0, 10])))

    return snake_pos

neutral_center = None
while True:
    win.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Detect faces in the frame
    faces = detector(frame, 1)

    for rect in faces:
        shape = predictor(frame, rect)

        if neutral_center is None:
            neutral_center = calculate_face_center(shape)
        else:
            movement = detect_face_movement(neutral_center, shape)
            if movement is not None:
                direction = movement

    snake_pos = update_snake_pos(snake_pos, direction)

    # Game over when snake is outside the screen
    if snake_pos[0][0] >= 650 or snake_pos[0][0] < 0 or snake_pos[0][1] >= 650 or snake_pos[0][1] < 0:
        game_over()

    draw_snake(snake_pos)
    pygame.display.update()
    pygame.time.Clock().tick(30)

cap.release()
cv2.destroyAllWindows()
