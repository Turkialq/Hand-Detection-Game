import math
import random

import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class Game:
    def __init__(self, path):
        self.points = []
        self.lengths = []  # distance between points
        self.currentLength = 0
        self.maxLength = 150  # fixed value and whenever we eat it will increase
        self.preHeadPoint = 0, 0
        self.score = 0
        self.gameOver = False

        self.foodImg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.foodImg.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

    def randomFoodLocation(self):
         self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgmain, currentHead):
        if self.gameOver:
            cvzone.putTextRect(imgmain, "Game Over", [300, 400],scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgmain, f'Your Score: {self.score}', [300, 550],scale=8, thickness=10, offset=20)

        else:
            cx, cy = currentHead
            px, py = self.preHeadPoint

            # reducing the tail

            if self.currentLength > self.maxLength:
                for i , length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.maxLength:
                        break

            # check if the snake ate the food

            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx <rx + self.wFood // 2 and ry - self.hFood // 2 < cy <ry + self.hFood:
                self.randomFoodLocation()
                self.maxLength += 50
                self.score += 1
                print(self.score)

            # Drawing the tails

            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.points.append([cx, cy])
            self.currentLength += distance
            self.preHeadPoint = cx, cy

            if self.points:
                for i, point in enumerate(self.points):
                    if i !=0:
                        cv2.line(imgmain, self.points[i-1], self.points[i], (0,0,150), 15)
                    cv2.circle(imgmain, self.points[-1], 15, (150,0,200),cv2.FILLED)

            # draw the food
            imgmain = cvzone.overlayPNG(imgmain, self.foodImg, (rx - self.wFood // 2, ry - self.hFood // 2))

            # Check for collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgmain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= minDist <= 1:
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        return imgmain


snakeGame = Game("Donut.png")


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType= False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = snakeGame.update(img, pointIndex)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


