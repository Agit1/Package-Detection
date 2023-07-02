# https://stackoverflow.com/questions/67302143/opencv-python-how-to-detect-filled-rectangular-shapes-on-picture
# Arturo L.
# 19-Dec-2022
# Version 1.2
import cv2
import math
import numpy as np

# Get the average bgr of a package
def getColor(box):

      # Get the average bgr of a diagonal made from box[0] and box[2]
      #print(box[0], " ", box[2])
      # With a step get total number of bgr pixels values
      step = 1
      absx = abs(box[0][0] - box[2][0])
      absy = abs(box[0][1] - box[2][1])
      if absx < absy:
            numberOfPoints = absx/step
      else:
            numberOfPoints = absy/step
      numberOfPoints = int(numberOfPoints)

      totalbgr = 0
      totalBlue = totalGreen = totalRed = 0

      newx = box[0][0]
      newy = box[0][1]

      for i in range(0, numberOfPoints):
            # positive x
            if box[0][0] > box[2][0]:
                  newx = newx - step
            else: # negative x
                  newx = newx + step

            # positive y
            if box[0][1] > box[2][1]:
                  newy = newy - step
            else: # negative y
                  newy = newy + step

            totalBlue = totalBlue + img[newx, newy][0]
            totalGreen = totalGreen + img[newx, newy][1]
            totalRed = totalRed + img[newx, newy][2]

            # Just for reference:
            # Draw point on diagonal, where we are taking the bgr
            print("totalbgr = ", img[newx, newy][0], img[newx, newy][1], img[newx, newy][2])
            #cv2.circle(img, (newx, newy), radius=3, color=(int(img[newx, newy][0]), int(img[newx, newy][1]), int(img[newx, newy][2])), thickness=-1)
            #cv2.circle(img, (newx, newy), radius=3, color=(int(img[newx, newy][0]), int(img[newx, newy][1]), int(img[newx, newy][2])), thickness=-1)
            cv2.circle(img, (newx, newy), radius=3, color=(0,0,255), thickness=-1)
      #totalbgr = totalbgr/(numberOfPoints)
      #totalbgr = int(totalbgr)
      print(totalbgr, totalBlue, totalGreen, totalRed)
      


src = '20221112_014332.jpg'
img = cv2.imread(src)

scale_percent = 18 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# adaptive threshold
#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,45)

cv2.imshow('thresh1', thresh)
# Fill rectangular contours
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

thresh = cv2.bitwise_not(thresh)
cv2.imshow('thresh2', thresh)
# Morph open
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Draw rectangles, the 'area_treshold' value was determined empirically
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area_treshold_min = 2000 # Package Area bigger than 2000, to prevent small noise-looking packages
area_treshold = 60000
for c in cnts:
    if cv2.contourArea(c) > area_treshold_min and cv2.contourArea(c) < area_treshold :
      x,y,w,h = cv2.boundingRect(c)
      #cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
      cv2.putText(img, 'Package', (-25+int((x+x+w)/2), int((y+y+h)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
      
      ## BEGIN - draw rotated rectangle
      rect = cv2.minAreaRect(c)
      box = cv2.boxPoints(rect)
      box = np.int0(box)


      # Version 1.2
      # Get the color of the package to be able to recognize between a box package or a plastic bag package
      bgr = img[300, 300]
      getColor(box)





      cv2.drawContours(img,[box],0,(52,235,235),2)
      ## END - draw rotated rectangle
      cv2.circle(img, (box[0][0],box[0][1]), radius=3, color=(0, 0, 255), thickness=-1)
      cv2.circle(img, (box[1][0],box[1][1]), radius=3, color=(0, 0, 255), thickness=-1)
      cv2.circle(img, (box[2][0],box[2][1]), radius=3, color=(0, 0, 255), thickness=-1)
      cv2.circle(img, (box[3][0],box[3][1]), radius=3, color=(0, 0, 255), thickness=-1)

      # Get orientation angle
      # First, lets get the lower point, which is the one with highest y coordinate
      lowestX, lowestY = 0, 0
      secondLowestX, secondLowestY = 0, 0
      for b in box:
      	if b[1:] > lowestY:
      		secondLowestX, secondLowestY = lowestX, lowestY
      		lowestX, lowestY = b
      	else:
      		if b[1:] > secondLowestY:
      			secondLowestX, secondLowestY = b

      #print(lowestX, lowestY)
      #print(secondLowestX, secondLowestY)
      #print(box)
      cv2.putText(img, 'P1', (lowestX, lowestY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      cv2.putText(img, 'P2', (secondLowestX, secondLowestY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      cv2.circle(img, (secondLowestX, lowestY), radius=3, color=(0, 0, 255), thickness=-1)
      cv2.putText(img, 'P3', (secondLowestX, lowestY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      # Draw reference line for orientation
      cv2.line(img, (lowestX, lowestY), (secondLowestX, lowestY), (252, 252, 3), 2)

      # Check if P3 is on the right or left of P1
      if (secondLowestX >= lowestX): # P2 on the right of P1
      	# theta = tg^-1 (((-1)*(YP2 - YP1))/(XP2 - XP1))
      	# theta = tg^-1 ((-1)*(secondLowestY - lowestY)/(secondLowestX - lowestX))
      	theta = math.degrees(math.atan((-1)*(secondLowestY - lowestY)/(secondLowestX - lowestX)))
      else: # P2 on the left of P1
        # theta = tg^-1 ((-1)*(YP2 - YP1)/(XP1 - XP2))
      	# theta = tg^-1 ((-1)*(secondLowestY - lowestY)/(lowestX - secondLowestX))
      	theta = math.degrees(math.atan((-1)*(secondLowestY - lowestY)/(lowestX - secondLowestX)))
      # Display orientation on degrees
      cv2.putText(img, str(round(theta, 2))+ " deg", (-25 +int((x+x+w)/2), int((y+y+h)/2)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

      
      



cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('image', img)
cv2.waitKey()


