import numpy as np
import cv2



"""Simple trials with images"""



# Read frame and display it
frame = cv2.imread('frame.png')
print('Image type:', type(frame), '\nArray dtype:', frame.dtype, '\nImage dimensions:', frame.shape)
print('Click on the image window and press any key to advance the program')
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyWindow('Frame')

# Define Mario template and display it
mario = frame[178:208, 112:128]
cv2.imshow('Mario', mario)
cv2.waitKey(0)
cv2.destroyWindow('Mario')

# Convert frame and mario to grayscale for use with matchTemplate()
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayMario = grayFrame[178:208, 112:128]

# Find the template according to the sum of square differences
method = cv2.TM_SQDIFF # can use TM_CCORR for cross-correlation, or normed methods with TM_***_NORMED
ssd = cv2.matchTemplate(grayFrame, grayMario, method=method)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(ssd) # we care about the minLoc for SQDIFF, maxLoc for CCORR

# Display the found template
mh, mw = mario.shape[:2] # for the height and width of the box we're drawing
x, y = minLoc # minLoc is a point (x, y)
cv2.rectangle(frame, (x, y), (x+mw, y+mh), color=[0,255,255], thickness=1) # draw rectangle
cv2.imshow('Found Template', frame)
cv2.waitKey(0)
cv2.destroyWindow('Found Template')



"""Simple trials with video"""



# Create a function for processing multiple frames
def trackTemplate(frame, templ):

    # convert frame and template to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayTempl = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    
    # match template
    ssd = cv2.matchTemplate(grayFrame, grayTempl, method=cv2.TM_SQDIFF)
    minLoc = cv2.minMaxLoc(ssd)[2]

    # draw box around template location
    th, tw = templ.shape[:2] # for the height and width of the box we're drawing
    x, y = minLoc # minLoc is a point (x, y)
    cv2.rectangle(frame, (x, y), (x+tw, y+th), color=[0,255,255], thickness=1) # draw rectangle
    
    return frame

# Create video capture object
cap = cv2.VideoCapture('smb.mp4')
print('Click on the image window and press [q] if you would like to quit before the video has finished.')

# Loop through video frames
while(cap.isOpened()):

    # Read a frame
    frame_exists, frame = cap.read()

    if frame_exists: 

        # Find the template
        frame = trackTemplate(frame, mario)

        cv2.imshow('Template Tracking',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # hold 'q' to quit the loop early
            break

    else:   
        break

cap.release()
cv2.destroyWindow('Template Tracking')



"""Getting better results with a mask"""



# Select just Marios head, create a mask removing sky-colored pixels
mario = mario[1:12,2:]
mask = np.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
])
cv2.imshow('Marios Head', mario)
cv2.waitKey(0)
cv2.destroyWindow('Marios Head')


# overload the trackTemplate() function to accept a mask
def trackTemplate(frame, templ, mask):
    
    # convert frame and template to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayTempl = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    
    # match template
    ssd = cv2.matchTemplate(grayFrame, grayTempl, method=cv2.TM_SQDIFF, mask=mask)
    minLoc = cv2.minMaxLoc(ssd)[2]

    # draw box around template location
    th, tw = templ.shape[:2] # for the height and width of the box we're drawing
    x, y = minLoc # minLoc is a point (x, y)
    cv2.rectangle(frame, (x-1, y), (x+tw+2, y+30), color=[0,255,255], thickness=1) # draw rectangle
    
    return frame

# Create video capture object
cap = cv2.VideoCapture('smb.mp4')

# Loop through video frames
while(cap.isOpened()):

    # Read a frame
    frame_exists, frame = cap.read()

    if frame_exists: 

        # Find the template
        frame = trackTemplate(frame, mario, mask)

        cv2.imshow('Template Tracking with mask',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # hold 'q' to quit the loop early
            break

    else:   
        break

cap.release()
cv2.destroyWindow('Template Tracking with mask')