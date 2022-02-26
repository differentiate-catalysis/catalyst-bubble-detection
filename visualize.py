import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def check_coalesce(center: List[int], radius: float, prev_circles: Dict, curr_circles: Dict, min_c: int = 10) -> bool:
    '''
    Performs a check of coalescence of bubbles between frames. Definition of coalescence:
    -Take two bubbles A and B at time (T-1) with radii r_A, r_B, respectively.
    -The two bubbles must be at a distance (r_A+r_B+min_c), otherwise coalescence not considered
    -At time step T, only one of bubbles A and B remain, and the radius of A has increased substantially.
    Parameters
    ----------
    center: list
        coordinates (x,y) for the center of a circle
    radius: float (or int)
        radius of circle in question
    prev_circles: dict
        dictionary containing all relevant info of past circles
    curr_circles: dict
        dictionary containing all relevant info of current circles
    min_c: int
        minimum distance between neighboring circles at which coalescence is deemed possible
    Returns
    -------
    is_coalesced: bool
        boolean where if True coalesence occurred, else False
    '''
    past_centers = np.array([np.array(prev_circles[i]['center']) for i in sorted(prev_circles.keys())])
    all_new_centers = np.array([np.array(curr_circles[i]['center']) for i in sorted(curr_circles.keys())])
    past_ind = np.argmin(np.sum((past_centers-np.array(center))**2,axis=1)**0.5)
    past_match = prev_circles[past_ind]
    poss_coalesce_inds = []
    for i in sorted(prev_circles.keys()):
        if i==past_ind: #Skip duplicates for past bubble frame
            continue
        poss_center = np.array(prev_circles[i]['center'])
        curr_center = past_match['center']
        if np.sum((poss_center-curr_center)**2)**0.5 < prev_circles[i]['radius']+past_match['radius']+min_c:
            poss_coalesce_inds.append(i) #Circles are close enough that they have potential to merge
    #For potential merge candidates, did any of them actually coalesce? If no match, then coalescenced occurred
    for i in poss_coalesce_inds:
        poss_center = np.array(prev_circles[i]['center'])
        new_center_dists = np.sum((all_new_centers-np.array(poss_center))**2,axis=1)**0.5
        if min(new_center_dists)>10: #No match to previous frame, so coalescence must have occurred
            return True #Coalescence
    return False #No coalescence


def compare_bubbles(center: List[int], radius: float, prev_circles: Dict, min_dist: int = 10) -> Tuple[bool, Optional[bool], Optional[float], Optional[int]]:
    '''
    Takes an existing bubble in a frame, determines whether the bubble is new, assessess volume change
    Parameters
    ----------
    center: list
        coordinates (x,y) for the center of a circle
    radius: float (or int)
        radius of circle in question
    prev_circles: dict
        dictionary containing all relevant info of past circles
    min_dist: int
        the minimum distance between circle centers at which one judges the formation of a new bubble
    Returns
    -------
    new: bool
        boolean stating whether a bubble is new in the current frame (True) or recurring (False)
    coalesced: bool (or None)
        boolean stating whether a bubble coalesced between frames (None if undetermined)
    volume_change: float (or None)
        change in volume of a bubble between slides (units are technically in px**3)
    old_id: int (or None)
        if bubble is not new, then keep the same old id between frames
    '''
    all_centers = np.array([np.array(prev_circles[i]['center']) for i in sorted(prev_circles.keys())])
    if len(all_centers)==0:
        new=True; coalesced=False; volume_change=None; old_id=None
        return new, coalesced, volume_change, old_id
    #Find matching circle based on min_dist parameter
    center_dists = np.sum((all_centers-np.array(center))**2,axis=1)**0.5
    if min(center_dists)<=10: #match located in previous frame
        matching_circle_data = prev_circles[np.argmin(center_dists)]
        new=False
        volume_change = 2.*np.pi / 3 * (radius**3 - matching_circle_data['radius']**3)
        if volume_change<=10: #Can edit threshold on what sort of volume change warrants coalescence
            coalesced=False
            return new, coalesced, volume_change, matching_circle_data['id']
        else:
            coalesced=None
            return new, coalesced, volume_change, matching_circle_data['id']
    else: #No match found, therefore circle is new
        new=True; coalesced=False; volume_change=None; old_id=None
        return new, coalesced, volume_change, old_id


def redraw_fig(frame: np.ndarray, bubble_dict: np.ndarray, color_dict: Dict = {'new':(224, 230, 73),'coalesced':(96, 66, 245),\
                  'growing':(50, 168, 82),'shrinking':(150, 0, 0),'unchanged':(179, 247, 255)}, frame_num: int = None) -> np.ndarray:
    '''
    Uses OpenCV functions on an existing frame to add circles based on category as well as id numbers
    Parameters
    ----------
    frame: ndarray
        image array of shape (x_len, y_len, 3) containing bubbles
    bubble_dict: dict
        dictionary containing all relevant info of current circles
    color_dict: dict
        dictionary containing all RGB colors for five modes (new, coalesced, growing, shrinking, unchanged)
    frame_num: int
        frame number to label imgae with
    Returns
    -------
    edit_frame: ndarray
        image array after cv2 has inserted circles and text at circle locations
    '''
    change_cutoff=50
    edit_frame = frame.copy()
    for b in bubble_dict.keys():
        bubble = bubble_dict[b]
        color=None
        if bubble['new']: color = color_dict['new']
        if bubble['coalesced']: color = color_dict['coalesced']
        if color is None: #New and coalesced categories take precedent over others
            if bubble['volume_change'] > change_cutoff: color = color_dict['growing']
            elif bubble['volume_change'] < -change_cutoff: color = color_dict['shrinking']
            else: color = color_dict['unchanged']
        cv2.circle(edit_frame, (bubble['center_x'], bubble['center_y']), int(bubble['radius']), color, 3)
        cv2.putText(edit_frame, str(bubble['id']), (bubble['center_x']-9, bubble['center_y']+7),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if frame_num:
            cv2.putText(edit_frame, str(frame_num), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return edit_frame


def label_image(image_file: str, npy_file: str, save_dir: str, bubble_color: Tuple[int, int, int] = (255,0,0), write_frame_csv: bool = False, save_image: bool = False) -> np.ndarray:
    '''
    Takes in an image file a npy file with bounding boxes, returns a single image with bubbles marked.
    Parameters
    ----------
    image_file: str
        name of image file to annotate
    npy_file: str
        name of .npy file containing bounding boxes corresponding to the image frame
    save_dir: str
        path to directory to write CSVs and images to
    bubble_color: tuple
        RGB color used to label bubbles in the image
    write_frame_csv: bool
        trigger to write a single csv file that describe all bubbles
    save_image: bool
        trigger to write a separate image file with circles and ids per bubble
    Returns
    -------
    edit_frame: ndarray
        final image after insertion of circles at marked bubble coordinates
    '''
    color_dict = {}
    for i in ['new','coalesced','growing','shrinking','unchanged']: color_dict[i]=bubble_color
    full_frame = cv2.imread(image_file)
    curr_boxes = np.load(npy_file)
    curr_circles={} #Tracking dict for all bubbles in frame
    circle_counter = 0
    image_dir = os.path.join(save_dir, 'labeled_images')
    csv_dir = os.path.join(save_dir, 'csv')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    for box in curr_boxes:
        curr_circles[circle_counter]={} #Tracking dict for current bubble
        center = [0.5*(box[0]+box[2]),0.5*(box[1]+box[3])] #Center averaging
        curr_circles[circle_counter]['center'] = center
        curr_circles[circle_counter]['center_x'] = int(center[0])
        curr_circles[circle_counter]['center_y'] = int(center[1])
        radius = 0.5*(abs(box[0]-center[0]) + abs(box[1]-center[1])) #Radius averaging
        curr_circles[circle_counter]['radius'] = radius
        curr_circles[circle_counter]['diameter'] = int(radius*2)
        curr_circles[circle_counter]['volume_change']=None
        curr_circles[circle_counter]['new']=True
        curr_circles[circle_counter]['coalesced']=False
        curr_circles[circle_counter]['id']=circle_counter
        circle_counter+=1
    if write_frame_csv:
        curr_df = pd.DataFrame.from_dict(curr_circles).T.drop(columns=['center',\
                                                    'radius']).sort_values(by='id')
        curr_df.to_csv(os.path.join(csv_dir, f'{os.path.basename(image_file)[:-4]}.csv'),index=False)
    edit_frame = redraw_fig(full_frame,curr_circles,color_dict=color_dict)
    if save_image:
        cv2.imwrite(os.path.join(image_dir, os.path.basename(image_file)),edit_frame)
    return edit_frame


def label_volume(movie_file: str, npy_folder: str, save_dir: str, write_frame_csv: bool = False, write_overall: bool = False, save_video: bool = False) -> pd.DataFrame:
    '''
    Takes in a movie file and all specified bounding boxes, returns descriptor files and edited movie.
    Parameters
    ----------
    movie_file: str
        path to the movie file to be parsed
    npy_folder: str
        folder name containing .npy files per frame of bounding boxes
    save_dir: str
        path to directory to write CSVs and images to
    write_frame_csv: bool
        trigger to write csv files at each frame that describe all bubbles
    write_overall: bool
        trigger to write a csv file that describes the overall movement and process of the bubbles
    save_video: bool
        trigger to write a separate video file with edited frames with circles and ids per bubble
    Returns
    -------
    overall_df: pd.DataFrame
        dataframe describing the full process of bubbles changing between frames
    '''
    capture = cv2.VideoCapture(movie_file)
    new_vid = None
    success = True
    prev_circles = {}
    total_volume_li = []
    num_bubbles_li = []
    max_id_num = 0
    frame = 0
    npy_files = sorted(np.array(os.listdir(npy_folder)).tolist())
    video_path = os.path.join(save_dir, os.path.basename(movie_file))
    csv_dir = os.path.join(save_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    while success: #Iterating over initial image frames, writing new frames to new video
        success,curr_frame = capture.read()
        if new_vid is None and save_video: #Open new video to begin writing, if desired
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            new_vid = cv2.VideoWriter(video_path,fourcc,60,\
                                       (int(capture.get(3)),int(capture.get(4))))
        if not success:
            break
        curr_boxes = np.load(os.path.join(npy_folder,npy_files[frame]))
        curr_circles = {} #Tracking dict for bubbles in current frame
        circle_counter = 0
        vol_in_frame = 0
        coalesce_check = []
        for box in curr_boxes:
            curr_circles[circle_counter]={} #Tracking dict for current bubble
            center = [0.5*(box[0]+box[2]),0.5*(box[1]+box[3])] #Center averaging
            curr_circles[circle_counter]['center'] = center
            curr_circles[circle_counter]['center_x'] = int(center[0])
            curr_circles[circle_counter]['center_y'] = int(center[1])
            radius = 0.5*(abs(box[0]-center[0]) + abs(box[1]-center[1])) #Radius averaging
            curr_circles[circle_counter]['radius'] = radius
            curr_circles[circle_counter]['diameter'] = int(radius*2)
            new, coalesced, volume_change, old_id = compare_bubbles(center,radius,prev_circles)
            if new:
                curr_circles[circle_counter]['id']=max_id_num
                max_id_num+=1
            else:
                curr_circles[circle_counter]['id']=old_id
            curr_circles[circle_counter]['new']=new
            curr_circles[circle_counter]['coalesced']=coalesced
            curr_circles[circle_counter]['volume_change']=volume_change
            vol_in_frame+=2.*np.pi/3. * radius**3 #Sum all circle volumes
            if coalesced is None:
                coalesce_check.append(circle_counter)
            circle_counter+=1
        #Check after for possible coalescences
        for i in coalesce_check:
            curr_circles[i]['coalesced']=check_coalesce(curr_circles[i]['center'],\
                                                        curr_circles[i]['radius'],\
                                                        curr_circles,prev_circles)
        total_volume_li.append(vol_in_frame) #Keep a list of all recorded volumes
        num_bubbles_li.append(int(len(curr_circles)))
        if len(curr_circles)>0:
            curr_df = pd.DataFrame.from_dict(curr_circles).T.drop(columns=['center',\
                                                                'radius']).sort_values(by='id')
        else:
            curr_df = None
        if save_video: #Write edited frame to new video
            new_vid.write(redraw_fig(curr_frame,curr_circles, frame_num=frame))
        if write_frame_csv and curr_df is not None:
            curr_df.to_csv(os.path.join(csv_dir, f'frame_{frame}.csv'),index=False)
        prev_circles = curr_circles
        frame+=1
    #Release videos when runs are done
    if save_video: new_vid.release()
    capture.release()
    overall_df = pd.DataFrame(np.array([total_volume_li,np.cumsum(total_volume_li).tolist(),\
                                        np.diff(total_volume_li,prepend=0.),num_bubbles_li]).T,\
                              columns=['frame_bubble_V','diff_bubble_V','total_bubble_V','num_bubbles'])
    if write_overall:
        overall_df.to_csv('overall_results.csv',index=False)
    return overall_df
