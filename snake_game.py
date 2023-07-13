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

def detect_face_movement(p1, p2):
    # Return the direction of face movement
    x_diff = p2.parts()[27].x - p1.parts()[27].x
    y_diff = p2.parts()[27].y - p1.parts()[27].y
    if x_diff > movement_threshold:
        return 'RIGHT'
    elif x_diff < -movement_threshold:
        return 'LEFT'
    elif y_diff > movement_threshold:
        return 'DOWN'
    elif y_diff < -movement_threshold:
        return 'UP'
    else:
        return ''

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

face_p1 = None
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

        if face_p1 is not None:
            movement = detect_face_movement(face_p1, shape)
            if movement:
                direction = movement

        face_p1 = shape

    snake_pos = update_snake_pos(snake_pos, direction)

    # Game over when snake is outside the screen
    if snake_pos[0][0] >= 600 or snake_pos[0][0] < 0 or snake_pos[0][1] >= 600 or snake_pos[0][1] < 0:
        game_over()

    draw_snake(snake_pos)
    pygame.display.update()
    pygame.time.Clock().tick(30)

cap.release()
cv2.destroyAllWindows()
