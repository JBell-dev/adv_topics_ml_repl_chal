import numpy as np
import cv2
import imageio
from os.path import dirname, realpath

filename = 'frames_2_4.npy'
scalingFactor = 8 # Note that the larger the scalingFactor, the longer the code takes to execute.
frameRate = 30 #fps

if __name__ == '__main__':
    try:
        assert filename[-4:] == '.npy'
        assert isinstance(scalingFactor, int) or isinstance(scalingFactor, float)
    except:
        raise TypeError('Wrong filetype, should be .npy')
    else:
        source_path = dirname(realpath(__file__))+'/np-data/'+filename
        output_path = dirname(realpath(__file__))+'/output/'+filename[:-4]+'.gif'
        frames = np.load(source_path).squeeze(-1)
        frame_list = []
        if scalingFactor != 1:
            scaled_frames = []
            for index in range(frames.shape[0]):
                scaled_frames.append(cv2.cvtColor(np.repeat(np.repeat(frames[index,:,:], scalingFactor, axis=0), scalingFactor, axis=1), cv2.COLOR_GRAY2RGB))
            frame_list = scaled_frames
        else:
            frame_list = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames]
        imageio.mimsave(output_path, frame_list, format="GIF", duration=1/frameRate)
        #imageio.imwrite(dirname(realpath(__file__))+'/output/exampleFrame.png', frame_list[252], format="PNG")