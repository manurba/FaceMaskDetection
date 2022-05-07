import cv2
import torch

from skvideo.io import FFmpegWriter, vreader

from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import numpy as np
import statistics

from train import MaskDetector
from common.facedetector import FaceDetector

'''
This module detects whether someone is wearing a mask or not whether it is
in real time streaming a camera connected to the PC or in a provided video.
'''

def tagVideo(modelpath, videopath, outputPath30fps, outputPath1fps, outputPath30fpsSmoothed):
    """
    Detect a person's face, place a rectangle around it and determine whether
    they are wearing a mask.

    The percentage accounts for the certainty of the prediction as per the model.
    """
    model = MaskDetector()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device)['state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    faceDetector = FaceDetector(
        prototype = 'models/deploy.txt',
        model = 'models/facedetect_300x300.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    if videopath is None:
        # Streams any camera available
        wid = 640
        hei = 480
        
        if outputPath30fps:
            # Save videos
            out = cv2.VideoWriter(outputPath30fps, -1, 30, (wid,hei))
            
            out1fps = cv2.VideoWriter(outputPath1fps, -1, 1, (wid,hei))

            out30fpsSmoothed = cv2.VideoWriter(outputPath30fpsSmoothed, -1, 30, (wid,hei))

        
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.namedWindow('30fps', cv2.WINDOW_NORMAL)
        cv2.namedWindow('1fps', cv2.WINDOW_NORMAL)
        cv2.namedWindow('30fps Smoothed', cv2.WINDOW_NORMAL)
        
        labels = ['No Mask', 'Mask']
        labelColor = [(0, 0, 204), (0, 153, 0)]

        # Stream from camera
        vid = cv2.VideoCapture(0)
        
        # Change window size
        vid.set(cv2.CAP_PROP_FRAME_WIDTH,wid)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT,hei)

        num_frames = 1

        pred = []
        aprobM = []
        countframe = 0

        while(True):
            # Continue streaming from the user's camera until the user presses 'q'
            for i in range(0, num_frames) :
                ret, frame = vid.read()

            #Operations on frame changing colours
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect all faces in the video in case there is more than one
            faces = faceDetector.detect(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            for face in faces:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                xStart, yStart, width, height = face
                
                # Fix coordinates that are outside of the image
                xStart, yStart = max(xStart, 0), max(yStart, 0)
                
                # Predict mask label on extracted face
                faceImg = frame[yStart:yStart+height, xStart:xStart+width]
                output = model(transformations(faceImg).unsqueeze(0).to(device))
        
                # Display probabilities under each of the faces detected
                _, predicted = torch.max(output.data, 1)
                sm = torch.nn.Softmax(dim=1)
                probabilities = sm(output)
                aprob = np.round_(torch.max(probabilities).cpu().detach().numpy()*100, decimals=2)
                aprobStr = str(aprob) + '%'
                

                # Change of colour
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw face frame
                cv2.rectangle(frame,
                            (xStart, yStart),
                            (xStart + width, yStart + height),
                            (126, 65, 64),
                            thickness=3)
                
                # Center text
                textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
                textX = xStart + width // 2 - textSize[0] // 2

                textSize1 = cv2.getTextSize(aprobStr, font, 1, 2)[0]
                textX1 = xStart + width // 2 - textSize1[0] // 2
                
                # Draw prediction label
                pred.append(predicted)
                aprobM.append(aprob)

                frame0 = frame.copy()

                cv2.putText(frame0,
                            labels[predicted],
                            (textX, yStart+height+35),
                            font, 1, labelColor[predicted], 2)
                
                cv2.putText(frame0,
                            aprobStr,
                            (textX1, yStart-20),
                            font, 1, labelColor[predicted], 2)


                if outputPath30fps:
                    out.write(frame0)
                cv2.imshow('30fps', frame0)

                if countframe >= 1:
                    frame2 = frame.copy()
                    cv2.putText(frame2,
                                labels[labelMode],
                                (textX, yStart+height+35),
                                font, 1, labelColor[labelMode], 2)
                
                    cv2.putText(frame2,
                                aprobMStr,
                                (textX1, yStart-20),
                                font, 1, labelColor[labelMode], 2)


                    if outputPath30fps:
                        out30fpsSmoothed.write(frame2)
                    cv2.imshow('30fps Smoothed', frame2)
                

                if countframe % 30 == 0 or countframe == 0:
                    frame1 = frame.copy()
                    
                    aprobMean = []
                    labelMode = []

                    labelMode = statistics.mode(pred)
                    aprobMean = np.round_(np.mean(aprobM),decimals=2)
                    aprobMStr = str(aprobMean) + '%'

                    cv2.putText(frame1,
                            labels[labelMode],
                            (textX, yStart+height+35),
                            font, 1, labelColor[labelMode], 2)
                
                    cv2.putText(frame1,
                            aprobMStr,
                            (textX1, yStart-20),
                            font, 1, labelColor[labelMode], 2)

                    if outputPath30fps:
                        out1fps.write(frame1)

                    cv2.imshow('1fps', frame1)

                    pred = []
                    aprobM = []
                    
                countframe += 1

            cv2.resizeWindow('30fps',wid,hei)
            # cv2.resizeWindow('1fps',wid,hei)
            # cv2.resizeWindow('30fps Smoothed',wid,hei)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        if outputPath30fps:
            out.release()
            out1fps.release()
            out30fpsSmoothed.release()

    else:
        outpath = 'output/video1.mp4'
        writer = FFmpegWriter(outpath)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.namedWindow('main', cv2.WINDOW_NORMAL)

        labels = ['No Mask', 'Mask']
        labelColor = [(0, 0, 204), (0, 153, 0)]

        for frame in vreader(str(videopath)):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = faceDetector.detect(frame)
            for face in faces:
                xStart, yStart, width, height = face
                
                # Fix coordinates that are outside of the image
                xStart, yStart = max(xStart, 0), max(yStart, 0)
                
                # Predict mask label on extracted face
                faceImg = frame[yStart:yStart+height, xStart:xStart+width]
                output = model(transformations(faceImg).unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)
                
                # Draw face frame
                cv2.rectangle(frame,
                            (xStart, yStart),
                            (xStart + width, yStart + height),
                            (126, 65, 64),
                            thickness=3)
                
                # Center text
                textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
                textX = xStart + width // 2 - textSize[0] // 2
                
                # Draw prediction label
                cv2.putText(frame,
                            labels[predicted],
                            (textX, yStart-20),
                            font, 1, labelColor[predicted], 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
 

    cv2.destroyAllWindows()


if __name__ == '__main__':
    modelpath = 'checkpoints/weights.ckpt'
    videopath = 'demo_input_vid\WIN_20201228_23_09_28_Pro.mp4'
    outputPath30fps = 'output/Test30fps.mp4'
    outputPath1fps = 'output/Test1fps.mp4'
    outputPath30fpsSmoothed = 'output/Test30fpsSmoothed.mp4'
    
    LiveRecord = 1

    if LiveRecord == 1:
        tagVideo(modelpath, None, outputPath30fps, outputPath1fps, outputPath30fpsSmoothed)
    else:
        # Not stable to use
        tagVideo(modelpath, videopath, None, None, None)